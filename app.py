from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3, os, joblib, numpy as np, datetime, bcrypt

DB_NAME = "health_data.db"
MODEL_PATH = "model.pkl"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-me-with-secure-random")

# --------------------------
# Database Setup
# --------------------------
def get_conn():
    return sqlite3.connect(DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password BLOB,
        age INTEGER DEFAULT 30,
        sex TEXT DEFAULT 'other',
        height REAL DEFAULT 170,
        weight REAL DEFAULT 70,
        sugar_drinks REAL DEFAULT 0,
        fruit_servings REAL DEFAULT 1,
        veg_servings REAL DEFAULT 2,
        stress_level REAL DEFAULT 5,
        systolic REAL DEFAULT 120,
        diastolic REAL DEFAULT 80,
        resting_hr REAL DEFAULT 70,
        existing_diabetes INTEGER DEFAULT 0,
        existing_cvd INTEGER DEFAULT 0,
        existing_cancer INTEGER DEFAULT 0,
        existing_asthma INTEGER DEFAULT 0,
        family_history_cancer INTEGER DEFAULT 0,
        family_history_cvd INTEGER DEFAULT 0,
        family_history_diabetes INTEGER DEFAULT 0
    )''')

    # Daily tracking data
    c.execute('''CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        date TEXT,
        sleep REAL,
        food REAL,
        exercise REAL,
        sugar_drinks REAL,
        fruit_servings REAL,
        veg_servings REAL,
        stress_level REAL
    )''')

    conn.commit()
    conn.close()
    print("✅ Database schema ensured")

init_db()

# --------------------------
# Load ML Model
# --------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("model.pkl not found. Run train_model.py first.")

model_bundle = joblib.load(MODEL_PATH)
scaler = model_bundle["scaler"]
models = model_bundle["models"]

# --------------------------
# Helper Functions
# --------------------------
def hash_pw(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_pw(password, hashed):
    return bcrypt.checkpw(password.encode("utf-8"), hashed)

def get_user(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    columns = [col[0] for col in c.description]
    conn.close()
    return dict(zip(columns, row)) if row else None

def to_float_or_default(val, default=None):
    try:
        return float(val) if val not in (None, "", "None") else default
    except:
        return default

def compute_features(user, avg_sleep=None, avg_food=None, avg_exercise=None):
    """Convert DB row to ML model feature vector with SI units."""
    sex_male = 1 if (str(user.get("sex", "")).lower() == "male") else 0
    height_m = to_float_or_default(user.get("height"), 1.70)
    weight_kg = to_float_or_default(user.get("weight"), 70)
    bmi = weight_kg / (height_m ** 2) if height_m else 24

    features = [
        to_float_or_default(user.get("age"), 35),
        sex_male,
        bmi,
        to_float_or_default(avg_sleep, 7),
        to_float_or_default(user.get("sugar_drinks"), 0.5),
        to_float_or_default(user.get("fruit_servings"), 1.0),
        to_float_or_default(user.get("veg_servings"), 2.0),
        to_float_or_default(avg_exercise, 30) * 100,
        to_float_or_default(user.get("stress_level"), 5),
        to_float_or_default(user.get("systolic"), 120),
        to_float_or_default(user.get("diastolic"), 80),
        to_float_or_default(user.get("resting_hr"), 70),
        int(user.get("existing_diabetes") or 0),
        int(user.get("existing_cvd") or 0),
        int(user.get("existing_cancer") or 0),
        int(user.get("existing_asthma") or 0),
        int(user.get("family_history_cancer") or 0),
        int(user.get("family_history_cvd") or 0),
        int(user.get("family_history_diabetes") or 0),
    ]
    return np.array([features])

def none_if_empty(v):
    return None if v in ("", None) else v

# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = get_conn()
        c = conn.cursor()
        try:
            c.execute("""INSERT INTO users (username,password) VALUES (?,?)""",
                      (username, hash_pw(password)))
            conn.commit()
            flash("Registration successful! Please login.", "info")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists", "error")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()

        if row and check_pw(password, row[0]):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("home"))

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    conn = get_conn()
    c = conn.cursor()

    if request.method == "POST":
        def get_checkbox(name): return 1 if request.form.get(name) == "on" else 0
        data = {
            "age": request.form.get("age"),
            "sex": request.form.get("sex"),
            "height": request.form.get("height"),
            "weight": request.form.get("weight"),
            "sugar_drinks": request.form.get("sugar_drinks"),
            "fruit_servings": request.form.get("fruit_servings"),
            "veg_servings": request.form.get("veg_servings"),
            "stress_level": request.form.get("stress_level"),
            "systolic": request.form.get("systolic"),
            "diastolic": request.form.get("diastolic"),
            "resting_hr": request.form.get("resting_hr"),
            "existing_diabetes": get_checkbox("existing_diabetes"),
            "existing_cvd": get_checkbox("existing_cvd"),
            "existing_cancer": get_checkbox("existing_cancer"),
            "existing_asthma": get_checkbox("existing_asthma"),
            "family_history_cancer": get_checkbox("family_history_cancer"),
            "family_history_cvd": get_checkbox("family_history_cvd"),
            "family_history_diabetes": get_checkbox("family_history_diabetes"),
        }

        c.execute("""UPDATE users SET
            age=?, sex=?, height=?, weight=?, sugar_drinks=?, fruit_servings=?, veg_servings=?,
            stress_level=?, systolic=?, diastolic=?, resting_hr=?, existing_diabetes=?, existing_cvd=?,
            existing_cancer=?, existing_asthma=?, family_history_cancer=?, family_history_cvd=?, family_history_diabetes=?
            WHERE username=?""",
            (
                data["age"], data["sex"], data["height"], data["weight"],
                data["sugar_drinks"], data["fruit_servings"], data["veg_servings"],
                data["stress_level"], data["systolic"], data["diastolic"],
                data["resting_hr"], data["existing_diabetes"], data["existing_cvd"],
                data["existing_cancer"], data["existing_asthma"],
                data["family_history_cancer"], data["family_history_cvd"], data["family_history_diabetes"],
                username
            ))
        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "info")
        return redirect(url_for("dashboard"))

    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return render_template("profile.html", user=user)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]

    if request.method == "POST" and "sleep" in request.form:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_conn()
        conn.execute("""INSERT INTO user_data 
            (username, date, sleep, food, exercise, sugar_drinks, fruit_servings, veg_servings, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                username, date,
                none_if_empty(request.form.get("sleep")),
                none_if_empty(request.form.get("food")),
                none_if_empty(request.form.get("exercise")),
                none_if_empty(request.form.get("sugar_drinks")),
                none_if_empty(request.form.get("fruit_servings")),
                none_if_empty(request.form.get("veg_servings")),
                none_if_empty(request.form.get("stress_level"))
            ))
        conn.commit()
        conn.close()
        flash("Daily data saved!", "info")
        return redirect(url_for("dashboard"))

    user = get_user(username)
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = conn.execute("""SELECT * FROM user_data WHERE username=? ORDER BY date""", (username,)).fetchall()
    conn.close()

    dates = [r["date"] for r in rows]
    sleep_data = [to_float_or_default(r["sleep"]) for r in rows]
    food_data = [to_float_or_default(r["food"]) for r in rows]
    exercise_data = [to_float_or_default(r["exercise"]) for r in rows]
    sugar_data = [to_float_or_default(r["sugar_drinks"]) for r in rows]
    fruit_data = [to_float_or_default(r["fruit_servings"]) for r in rows]
    veg_data = [to_float_or_default(r["veg_servings"]) for r in rows]
    stress_data = [to_float_or_default(r["stress_level"]) for r in rows]

    avg_sleep = np.nanmean([x for x in sleep_data if x is not None]) if sleep_data else None
    avg_food = np.nanmean([x for x in food_data if x is not None]) if food_data else None
    avg_exercise = np.nanmean([x for x in exercise_data if x is not None]) if exercise_data else None

    enough_days = len(rows) >= 7
    predictions = {}
    if user and enough_days:
        x = compute_features(user, avg_sleep, avg_food, avg_exercise)
        x_scaled = scaler.transform(x)
        for disease, bundle in models.items():
            clf = bundle["classifier"]
            proba = float(clf.predict_proba(x_scaled)[0][1])
            predictions[disease] = {
                "risk_pct": round(proba * 100, 2),
                "metrics": bundle["metrics"],
                "top_features": list(bundle["permutation_importance"].items())[:5]
            }

    chartJSON = {
        "dates": dates,
        "sleep": sleep_data,
        "food": food_data,
        "exercise": exercise_data,
        "sugar": sugar_data,
        "fruit": fruit_data,
        "veg": veg_data,
        "stress": stress_data
    }

    return render_template("dashboard.html", username=username,
                           chartJSON=chartJSON, predictions=predictions,
                           enough_days=enough_days, predictions_meta=user)

@app.route("/api/timeline")
def api_timeline():
    if "username" not in session:
        return jsonify({"error": "unauthorized"}), 401
    username = session["username"]
    conn = get_conn()
    rows = conn.execute("""SELECT date,sleep,food,exercise FROM user_data 
                        WHERE username=? ORDER BY date DESC LIMIT 30""", (username,)).fetchall()
    conn.close()
    timeline = [{"date": r[0], "sleep": r[1], "food": r[2], "exercise": r[3]} for r in rows]
    return jsonify({"timeline": timeline})

if __name__ == "__main__":
    app.run(debug=True)
