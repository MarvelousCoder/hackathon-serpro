var video;
var listinha = [];
let width = 640;
let height = 480;

function takeShot(action) {
  video.loadPixels();
  const myimage = video.canvas.toDataURL();
  const data = { myimage };
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide(); //hides video
  listinha.push({'action': action, 'img':data});
  // const response = fetch('/api', options);
  // const json = response.json();
  //console.log(json);
}

function mostra_resultado(item, index){
  $("#ul-resultado").append("<li><label>"+item['action']+"</label>: <p>"+item['resultado']+"</p></li>");
}

async function manda(){
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(listinha)
  };
  listinha = [];
  const response = await fetch('/api', options);
  const json = await response.json();
  console.log(json);

  json.forEach(mostra_resultado);
  //$("#lbl_nascimento").text(json['response']['dt_nascimento']);
  //$("#lbl_cpf").text(json['response']['cpf']);
  //const json = response.json();
}

function setup() {
  //noCanvas();
  createCanvas(640, 480); //canvas
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide(); //hides video

  // const button = document.getElementById('submit');
  // button.addEventListener('click', async event => {
  //   video.loadPixels();
  //   const myimage = video.canvas.toDataURL();
  //   const data = { myimage };
  //   video = createCapture(VIDEO);
  //   video.size(640, 480);
  //   video.hide(); //hides video
  //   const options = {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json'
  //     },
  //     body: JSON.stringify(data)
  //   };
  //   const response = await fetch('/api', options);
  //   const json = await response.json();
  //   console.log(json);

  // });
}

function draw() { //canvas
  //image(video, 0, 0, width, height); //video on canvas, position, dimensions
  translate(width,0); // move to far corner
  scale(-1.0,1.0);    // flip x-axis backwards
  image(video, 0, 0, width, height); //video on canvas, position, dimensions
}
