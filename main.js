const cvs = document.getElementById("cityCanvas");
const ctx = cvs.getContext("2d");
const img = new Image();
img.src = "./background.jpg";
img.onload = () => {
  ctx.drawImage(img, 0, 0);
};

// const bg = new Image();
// bg.src = "background.png";
// ctx.drawImage(bg, 10, 10);

// window.onload = function() {
//   var c = document.getElementById("cityCanvas");
//   var ctx = c.getContext("2d");
//   const bg = new Image();
//   bg.src = "background.jpg";
//   ctx.drawImage(bg, 10, 10);
// };

// window.addEventListener("DOMContentLoaded", function() {
//   // var image = document.getElementById("html5");
//   var canvas = document.getElementById("cityCanvas");
//   //   document.body.appendChild(canvas);

//   //   canvas.width = image.width;
//   //   canvas.height = image.height;

//   var context = canvas.getContext("2d");

//   const img = new Image();
//   img.src = "background.jpg";
//   context.drawImage(image, 0, 0);
// });
