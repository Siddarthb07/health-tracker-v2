document.addEventListener("DOMContentLoaded", function () {
  const data = window.__CHART_DATA__;
  if (!data || !data.dates || data.dates.length === 0) return;

  // History chart (sleep, food, exercise)
  const ctx1 = document.getElementById("historyChart").getContext("2d");
  new Chart(ctx1, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Sleep Hours",
          data: data.sleep,
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: false,
          tension: 0.3,
        },
        {
          label: "Calories",
          data: data.food,
          borderColor: "rgba(255, 159, 64, 1)",
          backgroundColor: "rgba(255, 159, 64, 0.2)",
          fill: false,
          tension: 0.3,
        },
        {
          label: "Exercise (Minutes)",
          data: data.exercise,
          borderColor: "rgba(54, 162, 235, 1)",
          backgroundColor: "rgba(54, 162, 235, 0.2)",
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: "Sleep, Calories, Exercise" } },
    },
  });

  // Lifestyle chart (sugar, fruit, veg, stress)
  const ctx2 = document.getElementById("lifestyleChart").getContext("2d");
  new Chart(ctx2, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Sugar Drinks (Servings)",
          data: data.sugar,
          borderColor: "rgba(255, 99, 132, 1)",
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          fill: false,
          tension: 0.3,
        },
        {
          label: "Fruit Servings",
          data: data.fruit,
          borderColor: "rgba(255, 206, 86, 1)",
          backgroundColor: "rgba(255, 206, 86, 0.2)",
          fill: false,
          tension: 0.3,
        },
        {
          label: "Veg Servings",
          data: data.veg,
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: false,
          tension: 0.3,
        },
        {
          label: "Stress Level",
          data: data.stress,
          borderColor: "rgba(153, 102, 255, 1)",
          backgroundColor: "rgba(153, 102, 255, 0.2)",
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: "Sugar, Stress, Fruits, Veggies" } },
    },
  });
});
