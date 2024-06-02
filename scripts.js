

function changeImage(element) {
    var mainImage = document.getElementById('displayed-image');
    mainImage.src = element.src;
}

const ctx = document.getElementById('chart');

new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple'],
      datasets: [{
        data: [12, 19, 3, 5, 2],
        borderWidth: 1,
      }]
    },
    
    options: {
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins:{
        legend: {
            display: false
        }
      },
      categoryPercentage: 0.85,
      barPercentage: 1.0,
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            display:false
            },
          border: {
            display: false,
          },
        },
        x: {
            grid: {
                display:false,
            },
            border: {
                display: false,
            },
            ticks: {
                display: false
            }
        },
      }
    }
});
  
