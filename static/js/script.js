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
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    shapes = JSON.parse(event.data);
    for (var a in shapes) {
      var obj = shapes[a];
      if (obj.id === "struct") {
        drawRectangle(obj.shape, obj.color);
      }
    }

    function drawRectangle(shape, color) {
      colors = getColor(color);
      ctx.beginPath();
      rectShape = rectifyAR(shape);
      ctx.rect(rectShape[0], rectShape[1], rectShape[2], rectShape[3]);
      ctx.fillStyle = colors;
      ctx.fill();
    }

    console.log(JSON.parse(event.data)[0].id);

    function rectifyAR(dims) {
      var wMult = window.innerWidth / 200;
      var hMult = window.innerHeight / 200;
      corrected_dims = [
        dims[0] * wMult,
        dims[1] * hMult,
        dims[2] * wMult,
        dims[3] * hMult
      ];
      return corrected_dims;
    }

    function getColor(arr) {
      return "rgb(" + arr[0] + "," + arr[1] + "," + arr[2] + ")";
    }
  });
});

window.addEventListener("resize", updateCanvas);
function updateCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
