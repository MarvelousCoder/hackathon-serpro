var video;
var width= 640;
var height = 480;

function setup() {
  //noCanvas();
  var canvas = createCanvas(320, 240); //canvas
  video = createCapture(VIDEO);
  video.size(320, 240);

  const button = document.getElementById('submit');
  button.addEventListener('click', async event => {
    video.loadPixels();
    const myimage = video.canvas.toDataURL();
    const data = { myimage };
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    };
    const response = await fetch('/read-cnh', options);
    const json = await response.json();
    console.log(json);
    //$("#lbl_nascimento").text(json['response']['dt_nascimento']);
    $("#lbl_cpf").text(json['response']['cpf']);
    //$("#lbl_nome").text(json['response']['nome']);
    video = createCapture(VIDEO);
    video.size(320, 240);
    video.hide(); //hides video
  });

  canvas.parent('sketch-holder');
  video.hide();
}

function draw() { //canvas
  //image(video, 0, 0, width, height); //video on canvas, position, dimensions
  translate(width,0); // move to far corner
  scale(-1.0,1.0);    // flip x-axis backwards
  image(video, 0, 0, width, height); //video on canvas, position, dimensions
}
