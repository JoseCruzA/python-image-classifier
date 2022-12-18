//Tomar y configurar el canvasC
let canvasC = document.getElementById("canvas");
let video = document.getElementById("video");
let othercanvas = document.getElementById("othercanvas");
let ctx = canvasC.getContext("2d");
let otherCtx = othercanvas.getContext("2d");
let size = 200;
let camaras = [];

let currentStream = null;
let facingMode = "user"; //Para que funcione con el celular (user/environment)

function mostrarCamara() {
  var opciones = {
    audio: false,
    video: {
      facingMode: "user",
      width: size,
      height: size,
    },
  };

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia(opciones)
      .then(function (stream) {
        currentStream = stream;
        video.srcObject = currentStream;
        procesarCamara();
        predecirC();
      })
      .catch(function (err) {
        alert("No se pudo utilizar la camara :(");
        console.log("No se pudo utilizar la camara :(", err);
        alert(err);
      });
  } else {
    alert(
      "No existe la funcion getUserMedia... oops :( no se puede usar la camara"
    );
  }
}

function cambiarCamara() {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => {
      track.stop();
    });
  }

  facingMode = facingMode == "user" ? "environment" : "user";

  var opciones = {
    audio: false,
    video: {
      facingMode: facingMode,
      width: size,
      height: size,
    },
  };

  navigator.mediaDevices
    .getUserMedia(opciones)
    .then(function (stream) {
      currentStream = stream;
      video.srcObject = currentStream;
    })
    .catch(function (err) {
      console.log("Oops, hubo un error", err);
    });
}

function predecirC() {
  if (modelo != null) {
    //Pasar canvasC a version 28x28
    resample_singleC(canvasC, 28, 28, othercanvas);

    var imgData = otherCtx.getImageData(0, 0, 28, 28);
    var arr = []; //El arreglo completo
    var arr28 = []; //Al llegar a arr150 posiciones se pone en 'arr' como un nuevo indice
    for (var p = 0, i = 0; p < imgData.data.length; p += 4) {
      var red = imgData.data[p] / 255;
      var green = imgData.data[p + 1] / 255;
      var blue = imgData.data[p + 2] / 255;
      arr28.push([red, green, blue]); //Agregar al arr150 y normalizar a 0-1. Aparte queda dentro de un arreglo en el indice 0... again
      if (arr28.length == 28) {
        arr.push(arr28);
        arr28 = [];
      }
    }

    arr = [arr]; //Meter el arreglo en otro arreglo por que si no tio tensorflow se enoja >:(
    //Nah basicamente Debe estar en un arreglo nuevo en el indice 0, por ser un tensor4d en forma 1, 28, 28, 1
    var tensor4 = tf.tensor4d(arr);
    var resultados = modelo.predict(tensor4).dataSync();
    var mayorIndice = resultados.indexOf(Math.max.apply(null, resultados));

    console.log("Prediccion", results[mayorIndice]);
    document.getElementById("resultado").innerHTML = results[mayorIndice];
  }

  setTimeout(predecir, 150);
}

function procesarCamara() {
  var ctx = canvasC.getContext("2d");

  ctx.drawImage(video, 0, 0, size, size, 0, 0, size, size);

  setTimeout(procesarCamara, 20);
}

/**
 * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
 *
 * @param {HtmlElement} canvasC
 * @param {int} width
 * @param {int} height
 * @param {boolean} resize_canvasC if true, canvasC will be resized. Optional.
 * Cambiado por RT, resize canvasC ahora es donde se pone el chiqitillllllo
 */
function resample_singleC(canvasC, width, height, resize_canvasC) {
  var width_source = canvasC.width;
  var height_source = canvasC.height;
  width = Math.round(width);
  height = Math.round(height);

  var ratio_w = width_source / width;
  var ratio_h = height_source / height;
  var ratio_w_half = Math.ceil(ratio_w / 2);
  var ratio_h_half = Math.ceil(ratio_h / 2);

  var ctx = canvasC.getContext("2d");
  var ctx2 = resize_canvasC.getContext("2d");
  var img = ctx.getImageData(0, 0, width_source, height_source);
  var img2 = ctx2.createImageData(width, height);
  var data = img.data;
  var data2 = img2.data;

  for (var j = 0; j < height; j++) {
    for (var i = 0; i < width; i++) {
      var x2 = (i + j * width) * 4;
      var weight = 0;
      var weights = 0;
      var weights_alpha = 0;
      var gx_r = 0;
      var gx_g = 0;
      var gx_b = 0;
      var gx_a = 0;
      var center_y = (j + 0.5) * ratio_h;
      var yy_start = Math.floor(j * ratio_h);
      var yy_stop = Math.ceil((j + 1) * ratio_h);
      for (var yy = yy_start; yy < yy_stop; yy++) {
        var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
        var center_x = (i + 0.5) * ratio_w;
        var w0 = dy * dy; //pre-calc part of w
        var xx_start = Math.floor(i * ratio_w);
        var xx_stop = Math.ceil((i + 1) * ratio_w);
        for (var xx = xx_start; xx < xx_stop; xx++) {
          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
          var w = Math.sqrt(w0 + dx * dx);
          if (w >= 1) {
            //pixel too far
            continue;
          }
          //hermite filter
          weight = 2 * w * w * w - 3 * w * w + 1;
          var pos_x = 4 * (xx + yy * width_source);
          //alpha
          gx_a += weight * data[pos_x + 3];
          weights_alpha += weight;
          //colors
          if (data[pos_x + 3] < 255) weight = (weight * data[pos_x + 3]) / 250;
          gx_r += weight * data[pos_x];
          gx_g += weight * data[pos_x + 1];
          gx_b += weight * data[pos_x + 2];
          weights += weight;
        }
      }
      data2[x2] = gx_r / weights;
      data2[x2 + 1] = gx_g / weights;
      data2[x2 + 2] = gx_b / weights;
      data2[x2 + 3] = gx_a / weights_alpha;
    }
  }

  for (var p = 0; p < data2.length; p += 4) {
    var gris = data2[p]; //Esta en blanco y negro

    if (gris < 100) {
      gris = 0; //exagerarlo
    } else {
      gris = 255; //al infinito
    }

    data2[p] = gris;
    data2[p + 1] = gris;
    data2[p + 2] = gris;
  }

  ctx2.putImageData(img2, 0, 0);
}
