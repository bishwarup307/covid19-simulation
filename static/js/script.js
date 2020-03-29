/*
 * __author__: bishwarup
 * created: Friday, 27th March 2020 3:43:50 am
 */
document.addEventListener("DOMContentLoaded", function(event) {
  const canvas = document.getElementById("canvas");

  /**@type {CanvasRenderingContext2D} */
  const ctx = canvas.getContext("2d");

  canvas.height = window.innerHeight;
  canvas.width = window.innerWidth;

  var eventSource = new EventSource("/stream");
  eventSource.addEventListener("update", function(event) {
    const canvas = document.getElementById("canvas");

    /** @type {CanvasRenderingContext2D} */
    const ctx = canvas.getContext("2d");
    function RenderContent() {
      // ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      shapes = JSON.parse(event.data);
      var images = [];
      var c = 0;

      for (var a in shapes) {
        var obj = shapes[a];
        console.log(obj);
        console.log(a);
        rectShape = rectifyAR(obj.shape);
        src = "../static/images/" + obj.id + ".png";
        l = Math.max(rectShape[2], rectShape[3]);

        loadAndDisplayImage(src, rectShape[0], rectShape[1], l, l);
      }
    }
    requestAnimationFrame(RenderContent);
    function loadAndDisplayImage(src, x, y, w, h) {
      var image;
      image = new Image();
      image.src = src;
      image.onload = function() {
        ctx.drawImage(this, x, y, w, h);
      };
    }

    function rectifyAR(dims) {
      var wMult = window.innerWidth / 200;
      var hMult = window.innerHeight / 200;
      corrected_dims = [
        Math.floor(dims[0] * wMult),
        Math.floor(dims[1] * hMult),
        Math.floor(dims[2] * wMult),
        Math.floor(dims[3] * hMult)
      ];
      return corrected_dims;
    }

    // function getColor(arr) {
    //   return "rgb(" + arr[0] + "," + arr[1] + "," + arr[2] + ")";
    // }
  });
});

window.addEventListener("resize", updateCanvas);
function updateCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
