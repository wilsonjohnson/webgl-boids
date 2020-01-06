// Import stylesheets
import './style.css';
import {mat4, vec2, vec4} from 'gl-matrix';
import {Boid, BoidManager} from './boids';
import {QuadTree} from './quadtree';

const {floor} = Math;
// import vertex_source from './vertex_shader.glsl';

const vertex_source = `
precision mediump float;
uniform float uTime;

attribute vec4 aVertexPosition;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

void main() {
  gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;
}
`;
const fragment_source = `
precision mediump float;
uniform float uTime;

float PI = 3.1415926535897;
float TAU = PI * 2.;

void main() {
  float r = sin( uTime ) + 1.;
  float g = sin( uTime + TAU / 3. ) + 1.;
  float b = sin( uTime + 2. * TAU / 3. ) + 1.;

  gl_FragColor = vec4(
    r,
    g,
    b,
    1.
  );
}
`;

type ShaderType = WebGLRenderingContextBase['FRAGMENT_SHADER']
  | WebGLRenderingContextBase['VERTEX_SHADER'];

function init_shader_program( gl: WebGLRenderingContext, vertex_source: string, fragment_source: string ): WebGLProgram {
  const vertex_shader = load_shader( gl, gl.VERTEX_SHADER, vertex_source );
  const fragment_shader = load_shader( gl, gl.FRAGMENT_SHADER, fragment_source );

  const shader_program = gl.createProgram();
  gl.attachShader(shader_program, vertex_shader);
  gl.attachShader(shader_program, fragment_shader);
  gl.linkProgram( shader_program );

  if (!gl.getProgramParameter(shader_program, gl.LINK_STATUS)) {
    throw 'Unable to initialize the shader program: ' + gl.getProgramInfoLog(shader_program);
  }

  return shader_program;
}

function load_shader( gl: WebGLRenderingContext, type: ShaderType, source: string ): WebGLShader {
  const shader = gl.createShader( type );

  gl.shaderSource( shader, source );
  gl.compileShader( shader );

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw 'An error occurred compiling the shaders: ' + info;
  }

  return shader;
}

function init_buffers( gl: WebGLRenderingContext ) {
  const position = gl.createBuffer();

  gl.bindBuffer( gl.ARRAY_BUFFER, position );

  const positions = [
    -0.75, -0.5,
    -0.75, 0.5,
     0.75, 0.0
  ];

  gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( positions ), gl.STATIC_DRAW );

  return { position };
}

function drawScene(gl: WebGLRenderingContext, program_info, buffers, timestamp: number ) {
  gl.clearColor(0.0, 0.0, 0.0, 1.0);  // Clear to black, fully opaque
  gl.clearDepth(1.0);                 // Clear everything
  gl.enable(gl.DEPTH_TEST);           // Enable depth testing
  gl.depthFunc(gl.LEQUAL);            // Near things obscure far things

  // Clear the canvas before we start drawing on it.

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Create a perspective matrix, a special matrix that is
  // used to simulate the distortion of perspective in a camera.
  // Our field of view is 45 degrees, with a width/height
  // ratio that matches the display size of the canvas
  // and we only want to see objects between 0.1 units
  // and 100 units away from the camera.

  const canvas = gl.canvas as HTMLCanvasElement;

  const FOV = 45 * Math.PI / 180;   // in radians
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const z_near = 0.1;
  const z_far = 100.0;
  const projection_matrix = mat4.create();

  // note: glmatrix.js always has the first argument
  // as the destination to receive the result.
  mat4.perspective(projection_matrix,
                   FOV,
                   aspect,
                   z_near,
                   z_far);


  // Set the drawing position to the "identity" point, which is
  // the center of the scene.
  const model_view_matrix = mat4.create();

  // Now move the drawing position a bit to where we want to
  // start drawing the square.
  const mat_copy = mat4.clone( model_view_matrix );

  
  mat4.translate(model_view_matrix,     // destination matrix
                 model_view_matrix,     // matrix to translate
                 [-0.0, 0.0, -6.0]);  // amount to translate
  mat4.rotateZ( model_view_matrix, model_view_matrix, timestamp/1000 );

  // Tell WebGL how to pull out the positions from the position
  // buffer into the vertexPosition attribute.
  {
    const num_components = 2;  // pull out 2 values per iteration
    const type = gl.FLOAT;    // the data in the buffer is 32bit floats
    const normalize = false;  // don't normalize
    const stride = 0;         // how many bytes to get from one set of values to the next
                              // 0 = use type and numComponents above
    const offset = 0;         // how many bytes inside the buffer to start from
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(
        program_info.attribLocations.vertexPosition,
        num_components,
        type,
        normalize,
        stride,
        offset);
    gl.enableVertexAttribArray(
        program_info.attribLocations.vertexPosition);
  }

  // Tell WebGL to use our program when drawing

  gl.useProgram(program_info.program);

  // Set the shader uniforms

  gl.uniformMatrix4fv(
      program_info.uniformLocations.projectionMatrix,
      false,
      projection_matrix);
  gl.uniformMatrix4fv(
      program_info.uniformLocations.modelViewMatrix,
      false,
      model_view_matrix);

  const uTime = timestamp / 1000;
  gl.uniform1f(
    program_info.uniformLocations.timestamp,
    uTime
  );
  document.getElementById('display').innerText = `${uTime}`;

  {
    const offset = 0;
    const vertexCount = 3;
    gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);
  }
}

// Write TypeScript code!
const appDiv: HTMLElement = document.getElementById('app');
appDiv.innerHTML = `
<div id="display"></div>
<canvas id="canvas"
        style="margin: 0;
        padding: 0;
        display: inline-flex;
        position: absolute;
        top: 25px;
        left: 0px;">
</canvas>
<canvas id="other"
        style="margin: 0;
        padding: 0;
        display: inline-flex;
        position: absolute;
        top: 525px;
        left: 0px;">
</canvas>
`;

const canvas: HTMLCanvasElement = document.getElementById('canvas') as HTMLCanvasElement;
const other: HTMLCanvasElement = document.getElementById('other') as HTMLCanvasElement;
canvas.width = 500;
canvas.height = 500;
other.width = 500;
other.height = 500;
console.log( vertex_source );

const gl: WebGL2RenderingContext = canvas.getContext('webgl2');

const program = init_shader_program( gl, vertex_source, fragment_source );

const programInfo = {
  program,
  attribLocations: {
    vertexPosition: gl.getAttribLocation(program, 'aVertexPosition'),
  },
  uniformLocations: {
    projectionMatrix: gl.getUniformLocation(program, 'uProjectionMatrix'),
    modelViewMatrix: gl.getUniformLocation(program, 'uModelViewMatrix'),
    timestamp: gl.getUniformLocation(program, 'uTime'),
  },
};

const buffers = init_buffers( gl );

const boids = [];

let last_x;
let last_y;

// generate boids randomly across the entire canvas
function type_rand(x= 10, y=100) {
  for( let i = 0; i< x; i++ ){
    for( let j = 0; j< y; j++ ){
      const velocity = vec2.random(vec2.create());
      const position = vec2.add( vec2.create(),
        vec2.mul(
          vec2.create(),
          vec2.fromValues( Math.random(), Math.random() ),
          vec2.fromValues( gl.canvas.width, gl.canvas.height ),
        ),
        vec2.fromValues( gl.canvas.width / 2, gl.canvas.height / 2 )
      );

      boids.push(new Boid( canvas, position, velocity ));
      last_x = i * 5;
      last_y = j * 5;
    }
  }
}

// generates a boid doughnut around canvas center
function type1() {
  for( let i = 0; i< 10; i++ ){
    for( let j = 0; j< 100; j++ ){
      const velocity = vec2.fromValues(0,1);
      const position = vec2.add( vec2.create(),vec2.scale( vec2.create(),vec2.random( vec2.create() ), Math.random() * 100 + 150), vec2.fromValues( gl.canvas.width / 2, gl.canvas.height / 2 ));
      boids.push(new Boid( canvas, position, velocity ));
      last_x = i * 5;
      last_y = j * 5;
    }
  }
}

// generates a spiral of boids from canvas center
function type2(num = 1000) {
  let mag = 10;
  const start = vec2.random(vec2.create());
  for( let i = 0; i< num; i++ ){
      const velocity = vec2.fromValues(0,1);
      const position = vec2.scale( vec2.create(), vec2.rotate( vec2.create(), start, vec2.create(), i * Math.E * 2 ), mag );
      mag = mag + 0.05 * Math.E;
      vec2.add( position, position ,vec2.fromValues( gl.canvas.width / 2, gl.canvas.height / 2 ));
      boids.push(new Boid( canvas, position, velocity ));
      last_x = position[0];
      last_y = position[1];
  }
}

// type_rand(10, 10);
// type1();
type2(200);


let frametime = 0;
let last = 0;
const rates = [];

const _ = () => vec2.create();
const dimensions = vec4.fromValues( 0, 0, gl.canvas.width, gl.canvas.height );

const boid_manager = new BoidManager(
  boids,
  dimensions
);

const USE_MANAGER = false;

function render( timestamp: number ) {
  // drawScene( gl, programInfo, buffers, timestamp );
  const in_seconds = timestamp * 0.001;
  frametime = in_seconds - last;
  last = in_seconds;
  rates.push( 1/frametime )
  if(rates.length > 40) {
    rates.shift();
  }
  const framerate = rates.reduce((a,c)=>a+c,0)/rates.length;
  // if ( timestamp % 100 && framerate > 55 ) {
  //   const last = boids[boids.length-1];

  //   const x = last_x + 5 >= gl.canvas.width
  //     ? 0
  //     : last_x + 5;
  //   const y = x === 0 ? last_y + 5: last_y;
  //   last_x = x;
  //   last_y = y;
  //   boids.push(new Boid( canvas, x, y ));
  // }
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  const GL = gl;
  GL.clearColor(0.0, 0.0, 0.0, 1.0);  // Clear to black, fully opaque
  GL.clearDepth(1.0);                 // Clear everything
  GL.enable(GL.DEPTH_TEST);           // Enable depth testing
  GL.depthFunc(GL.LEQUAL);            // Near things obscure far things

  // Clear the canvas before we start drawing on it.

  GL.clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT);
  
  
  let quadtree;
  if ( USE_MANAGER ) {
    quadtree = boid_manager.flock_and_render( timestamp, frametime );
  } else {
    let t;
    quadtree = new QuadTree(boids, dimensions, i=>i.position, 10 );
    for ( let boid of boids ) {
      const tr = quadtree.query_circle([boid.x, boid.y, 35]);
      if ( ! t ) t = tr;
      boid.hue = tr.length * boids.length / Math.E;
      boid.flock( tr, 60 * frametime );
    }
    for ( let boid of boids ) {
      boid.update(60 * frametime);
      boid.render();
    }
  }

  document.getElementById('display').innerText = `${floor(boids[0].x)},${floor(boids[0].y)}@${quadtree.length} => ${floor(framerate)}@${frametime}`;
  other.getContext('2d').drawImage( canvas, 0, 0);
  quadtree.draw( other.getContext('2d') );
  
  window.requestAnimationFrame(render);
  // setTimeout( () => window.requestAnimationFrame(render), 1000 );
}

window.requestAnimationFrame(render);

// console.log(  );
