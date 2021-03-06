import {mat4, vec2, vec4} from 'gl-matrix';
import {QuadTree} from '../quadtree';


let GL: WebGL2RenderingContext;
const boid_vert_source = `
precision mediump float;
uniform float uTime;

attribute vec2 a_position;

uniform vec2 u_resolution;
uniform vec2 u_translation;
uniform vec2 u_rotation;
uniform vec2 u_scale;

uniform vec2 u_camera_position;
uniform vec2 u_camera_scale;
uniform vec2 u_camera_rotation;

vec2 apply_camera( vec2 position, vec2 scale, vec2 rotation, vec2 translation ) {
  // Scale the position
  vec2 scaledPosition = position * scale;

  // Rotate the position
  vec2 rotatedPosition = vec2(
     scaledPosition.x * rotation.y + scaledPosition.y * rotation.x,
     scaledPosition.y * rotation.y - scaledPosition.x * rotation.x);

  // Add in the translation.
  return rotatedPosition + translation;
}

vec2 apply_camera_to_clipspace( vec2 clipspace, vec2 resolution, vec2 scale, vec2 rotation, vec2 translation ) {
  vec2 position = clipspace * resolution;
  position = apply_camera( position, scale, rotation, translation );
  return position / resolution;
}

vec2 create_view( vec2 position, vec2 scale, vec2 rotation, vec2 translation ) {
  // Scale the position
  vec2 scaledPosition = position * scale;

  // Rotate the position
  vec2 rotatedPosition = vec2(
     scaledPosition.x * rotation.y + scaledPosition.y * rotation.x,
     scaledPosition.y * rotation.y - scaledPosition.x * rotation.x);

  // Add in the translation.
  return rotatedPosition + translation;
}

vec2 clip( vec2 position, vec2 resolution ) {
  // convert the position from pixels to 0.0 to 1.0
  vec2 zeroToOne = position / resolution;

  // convert from 0->1 to 0->2
  vec2 zeroToTwo = zeroToOne * 2.0;

  // convert from 0->2 to -1->+1 (clipspace)
  return zeroToTwo - 1.0;
}

void main() {
  vec2 position = create_view(a_position, u_scale, u_rotation, u_translation);

  vec2 clipSpace = clip( position, u_resolution );

  clipSpace = apply_camera_to_clipspace(clipSpace, u_resolution, u_camera_scale, u_camera_rotation, u_camera_position);

  gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
}
`;
const boid_frag_source = `
precision mediump float;
uniform float uTime;

float PI = 3.1415926535897;
float HALF_PI = PI * .5;
float TAU = PI * 2.;

void main() {
  float r = clamp(HALF_PI *sin( uTime ) , .0, 1. );
  float g = clamp(HALF_PI *sin( uTime + TAU / 3.) , .0, 1. );
  float b = clamp(HALF_PI *sin( uTime + 2. * TAU / 3.) , .0, 1. );

  gl_FragColor = vec4(
    r,
    g,
    b,
    1.
  );
}
`;

type Vec2Array = [number, number];

export class Torus {
  constructor(
    public readonly dimensions: vec2
  ) {
  }

  public offset( from: vec2, to: vec2 ): vec2 {
    return Torus.offset( this.dimensions, from, to );
  }
  
  public static offset( dimensions: vec2, from: vec2, to: vec2 ): vec2 {
    let delta = vec2.subtract( vec2.create(), to, from );
    let abs = vec2.create();
    abs[0] = Math.abs(delta[0]);
    abs[1] = Math.abs(delta[1]);

    if ( abs[0] > dimensions[0] / 2 ) delta[0] = -Math.sign(delta[0]) * (dimensions[0] - abs[0]);
    if ( abs[1] > dimensions[1] / 2 ) delta[1] = -Math.sign(delta[1]) * (dimensions[1] - abs[1]);

    return delta;
  }

  public distance_squared( from: vec2, to: vec2 ): number {
    return Torus.distance_squared( this.dimensions, from, to );
  }

  public distance( from: vec2, to: vec2 ): number {
    return Torus.distance( this.dimensions, from, to );
  }

  public static distance_squared( dimensions: vec2, from: vec2, to: vec2 ): number {
    let delta = vec2.subtract( vec2.create(), to, from.map(Math.abs) as vec2 );

    if ( delta[0] > dimensions[0] / 2 ) delta[0] = dimensions[0] - delta[0];
    if ( delta[1] > dimensions[1] / 2 ) delta[1] = dimensions[1] - delta[1];

    return vec2.sqrLen(delta);
  }

  public static distance( dimensions: vec2, from: vec2, to: vec2 ): number {
    return Math.sqrt(Torus.distance_squared( dimensions, from, to ));
  }
}

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

function load_shader( gl: WebGLRenderingContext, type, source: string ): WebGLShader {
  const shader = gl.createShader( type );

  gl.shaderSource( shader, source );
  gl.compileShader( shader );

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw 'An error occurred compiling the shaders:\n' + info;
  }

  return shader;
}

function limit( out: vec2, vector: vec2, magnitude: number ): vec2 {
  const magnitude_squared = magnitude ** 2;

  return out;
}

function get_constraints(): vec2 {
  return vec2.fromValues( GL.canvas.width, GL.canvas.height );
}

export class Camera2D {
  private u_camera_position: WebGLUniformLocation;
  private u_camera_scale: WebGLUniformLocation;
  private u_camera_rotation: WebGLUniformLocation;

  private position: vec2 = vec2.create();
  private scale: vec2 = vec2.fromValues(1,1);
  private rotation: vec2 = vec2.create();

  constructor(
    private program: WebGLProgram
  ) {
    this.u_camera_position = GL.getUniformLocation(this.program, 'u_camera_position');
    this.u_camera_scale = GL.getUniformLocation(this.program, 'u_camera_scale');
    this.u_camera_rotation = GL.getUniformLocation(this.program, 'u_camera_rotation');
    this.angle = Math.PI;
  }

  public update() {
    GL.uniform2fv(this.u_camera_position, this.position);
    GL.uniform2fv(this.u_camera_scale, this.scale);
    vec2.normalize( this.rotation, this.rotation );
    GL.uniform2fv(this.u_camera_rotation, this.rotation);
  }

  public set translation( value: vec2 ) {
    this.position = value;
  }

  public get translation() {
    return this.position;
  }

  public set angle( value: number ) {
    this.rotation = vec2.rotate(
      vec2.create(),
      vec2.fromValues(1,0),
      vec2.create(),
      value
    );
  }
}

export class Boid {
  private static program: WebGLProgram;
  private a_position: number;
  private u_timestamp: WebGLUniformLocation;
  private u_scale: WebGLUniformLocation;
  private u_translation: WebGLUniformLocation;
  private u_resolution: WebGLUniformLocation;
  private u_rotation: WebGLUniformLocation;
  public static camera: Camera2D;
  private static position: WebGLBuffer;
  private acceleration: vec2;
  private _hue: number;
  private personal_bubble: number;
  private _max_acceleration: number;
  private _max_acceleration_squared: number;
  
  constructor(
    canvas: HTMLCanvasElement,
    public position: vec2,
    public velocity: vec2
  ) {
    if ( !GL ) GL = canvas.getContext('webgl2');
    if (!Boid.program) Boid.program = init_shader_program( GL, boid_vert_source, boid_frag_source );
    this.a_position = GL.getAttribLocation(Boid.program, 'a_position');
    this.u_timestamp = GL.getUniformLocation(Boid.program, 'uTime');
    this.u_scale = GL.getUniformLocation(Boid.program, 'u_scale');
    this.u_translation = GL.getUniformLocation(Boid.program, 'u_translation');
    this.u_resolution = GL.getUniformLocation(Boid.program, 'u_resolution');
    this.u_rotation = GL.getUniformLocation(Boid.program, 'u_rotation');
    if (!Boid.camera) Boid.camera = new Camera2D( Boid.program );
    this.acceleration = vec2.fromValues( 0, 0 );
    this.personal_bubble = 25;
    this.max_acceleration = 5;

    if ( ! this.position ) this.position = vec2.fromValues( 0, 0 );
    if ( ! this.velocity ) this.velocity = vec2.fromValues( 1, 0 );
    if ( ! Boid.position ) {
    
      Boid.position = GL.createBuffer();

      GL.bindBuffer( GL.ARRAY_BUFFER, Boid.position );
      const positions = [
        -5, -2.5,
        -5, 2.5,
        5, 0.0
      ];

      GL.bufferData( GL.ARRAY_BUFFER, new Float32Array( positions ), GL.STATIC_DRAW );
    }
  }

  public get max_acceleration(): number {
    return this._max_acceleration;
  }

  public set max_acceleration( value: number ) {
    this._max_acceleration = value;
    this._max_acceleration_squared = value ** 2;
  }

  public get max_acceleration_squared(): number {
    return this._max_acceleration_squared;
  }

  public set max_acceleration_squared( value: number ) {
    this._max_acceleration_squared = value;
    this._max_acceleration = Math.sqrt( value );
  }

  public limit_acceleration( ) {
    if ( vec2.squaredLength(this.acceleration)> this.max_acceleration_squared ) {
      vec2.normalize(this.acceleration, this.acceleration);
      vec2.scale(this.acceleration, this.acceleration ,this.max_acceleration)
    }
  }

  public add_force( force: vec2 ) {
    vec2.add( this.acceleration, this.acceleration, force );
  }

  private update_alignment( others: Boid[], delta: number ) {
    if ( others.length === 0 ) return;
    const target = vec2.create();
    for( let other of others ) {
      vec2.add( target, target, other.velocity );
    }
    vec2.scale( target, target,  1 / ( others.length ) );
    vec2.subtract( target, target, this.velocity);
    vec2.scale( target, target,  delta/8 );

    this.add_force( target );
  }

  private update_grouping( others: Boid[], delta: number ) {
    if ( others.length === 0 ) return;
    const vector: vec2 = vec2.create();
    for ( let other of others ) {
      vec2.add( vector, vector, Torus.offset( get_constraints(), this.position, other.position ) );
    }
    vec2.scale(vector, vector, 1 / others.length );
    // vector.subtract( this.position );
    vec2.scale(vector, vector, delta / 100 );
    this.add_force( vector );
  }

  private update_separation( others: Boid[], delta: number ) {
    if ( others.length === 0 ) return;
    const target = vec2.create();
    for( let other of others ) {
      if ( other === this ) continue;
      const between: vec2 = Torus.offset( get_constraints(), this.position, other.position );
      const length = vec2.length(between);
      if ( length > 0 && length < this.personal_bubble ) {
        vec2.scale( between, between, 1 / length );
        vec2.subtract( target, target, between );
      }
    }
    vec2.scale( target, target, delta / others.length );
    this.add_force( target );
  }

  public flock( others: Boid[], delta: number ) {
    this.update_alignment( others, delta );
    this.update_grouping( others, delta );
    this.update_separation( others, delta );
  }

  public update( delta: number = 1 ) {
    vec2.add( this.velocity, this.velocity, vec2.scale( vec2.create(), this.acceleration, delta ) );
    if ( vec2.sqrLen(this.velocity) > 4 ) {
      vec2.normalize( this.velocity, this.velocity );
      vec2.scale( this.velocity, this.velocity, 2 );
    }
    vec2.scale( this.velocity, this.velocity, delta );

    vec2.add( this.position, this.position, this.velocity );
    if ( this.position[0] > GL.canvas.width ) this.position[0] = this.position[0] - GL.canvas.width; 
    if ( this.position[0] < 0 ) this.position[0] = this.position[0] + GL.canvas.width; 
    if ( this.position[1] > GL.canvas.height ) this.position[1] = this.position[1] - GL.canvas.height; 
    if ( this.position[1] < 0 ) this.position[1] = this.position[1] + GL.canvas.height; 
  }

  public get x() { return this.position[0] };
  public get y() { return this.position[1] };

  public set hue( value: number ) {
    this._hue = value;
  }

  public render( timestamp: number = 0 ) {

    // Create a perspective matrix, a special matrix that is
    // used to simulate the distortion of perspective in a camera.
    // Our field of view is 45 degrees, with a width/height
    // ratio that matches the display size of the canvas
    // and we only want to see objects between 0.1 units
    // and 100 units away from the camera.

    const canvas = GL.canvas as HTMLCanvasElement;    

    // Tell WebGL how to pull out the positions from the position
    // buffer into the vertexPosition attribute.
    {
      const num_components = 2;  // pull out 2 values per iteration
      const type = GL.FLOAT;    // the data in the buffer is 32bit floats
      const normalize = false;  // don't normalize
      const stride = 0;         // how many bytes to get from one set of values to the next
                                // 0 = use type and numComponents above
      const offset = 0;         // how many bytes inside the buffer to start from
      GL.bindBuffer(GL.ARRAY_BUFFER, Boid.position);
      GL.vertexAttribPointer(
          this.a_position,
          num_components,
          type,
          normalize,
          stride,
          offset);
      GL.enableVertexAttribArray(this.a_position);
    }

    // Tell WebGL to use our program when drawing

    GL.useProgram(Boid.program);

    // Set the shader uniforms
    GL.uniform2f(this.u_scale,1,1);
    GL.uniform2fv(this.u_translation, this.position);
    GL.uniform2f(this.u_resolution, GL.canvas.width, GL.canvas.height);

    Boid.camera.update();

    const rotation = vec2.normalize( vec2.create(), this.velocity );

    GL.uniform2fv(this.u_rotation, vec2.rotate(rotation, rotation, vec2.fromValues(0,0), Math.PI/2));

    const uTime = ( timestamp + this._hue ) / 1000;
    GL.uniform1f(
      this.u_timestamp,
      uTime
    );

    {
      const offset = 0;
      const vertexCount = 3;
      GL.drawArrays(GL.TRIANGLE_STRIP, offset, vertexCount);
    }
  }
}

const boid_manager_vert_source = `#version 300 es
precision mediump float;
uniform float uTime;

in vec2 a_position;

uniform vec2 u_resolution;
uniform vec2 u_translation;
uniform vec2 u_rotation;
uniform vec2 u_scale;

uniform vec2 u_camera_position;
uniform vec2 u_camera_scale;
uniform vec2 u_camera_rotation;

out vec2 v_Position;

vec4 vecOut( vec2 from ) {
  return vec4( from, 0, 1 );
}

void main() {
  // gl_Position = vecOut(a_position);
  // Scale the position
  vec2 scaledPosition = a_position * u_scale;
  // gl_Position = vecOut( a_position * u_scale );

  // Rotate the position
  vec2 rotatedPosition = vec2(
     scaledPosition.x * u_rotation.y + scaledPosition.y * u_rotation.x,
     scaledPosition.y * u_rotation.y - scaledPosition.x * u_rotation.x);
  // gl_Position = vecOut(vec2(
  //    scaledPosition.x * u_rotation.y + scaledPosition.y * u_rotation.x,
  //    scaledPosition.y * u_rotation.y - scaledPosition.x * u_rotation.x));

  // Add in the translation.
  vec2 position = rotatedPosition + u_translation;
  // gl_Position = vecOut( rotatedPosition + u_translation );

  // convert the position from pixels to 0.0 to 1.0
  vec2 zeroToOne = position / u_resolution;
  // gl_Position = vecOut( position / u_resolution );

  // convert from 0->1 to 0->2
  vec2 zeroToTwo = zeroToOne * 2.0;
  // gl_Position = vecOut( zeroToOne * 2.0 );

  // convert from 0->2 to -1->+1 (clipspace)
  vec2 clipSpace = zeroToTwo - 1.0;
  gl_Position = vecOut( zeroToTwo - 1.0 );

  // gl_Position = vecOut(clipSpace * vec2(1, -1));
}
`;
const boid_manager_frag_source = `#version 300 es
#define MAX_POSITIONS_LENGTH 1000
#define PI 3.1415926535897

precision mediump float;

struct Boid {
  vec2 position;
  vec2 velocity;
};

layout( std140 ) uniform u_boids {
  Boid boids[MAX_POSITIONS_LENGTH];
};

uniform float uTime;
uniform vec2 u_resolution;

float HALF_PI = PI * .5;
float TAU = PI * 2.;

out vec4 fragColor;

float LineTest( vec2 v1, vec2 v2, vec2 p ) {
  return (p.x - v1.x) * (v2.y - v1.y) - (p.y - v1.y) * (v2.x-v1.x);
}

vec4 PaintBoid( vec2 uv, Boid b ) {
  vec2 offset = b.position + b.velocity;
  vec4 color = vec4(0,0,0,1);
  vec2 vel = normalize(b.velocity) * 3.;
  vec2 left = vel.yx * vec2(1.,-1.);
  vec2 right = vel.yx * vec2(-1.,1.);
  left = left - vel + b.position;
  right = right - vel + b.position;
  vec2 point = vel * 2. + b.position;
  if ( LineTest(left, right, uv) >= 0. &&
    LineTest(left, point, uv) <= 0. &&
    LineTest(right, point, uv) >= 0.
  ) // color += vec4( smoothstep(vec2(0),u_resolution,uv), 0, 1);
  color += vec4( smoothstep(vec2(0),u_resolution,uv) * normalize( vel ), 1, 1);
  // if ( distance( uv, b.velocity + b.position ) < 3. ) color += vec4( 0,1,0,0 );
  // if ( distance( uv, b.position ) < 5. ) 
  return color;
}

void main() {
  vec2 uv = (gl_FragCoord.xy * .5 * u_resolution.xy) / u_resolution.y;
  float r = clamp(HALF_PI *sin( uTime ) , .0, 1. );
  float g = clamp(HALF_PI *sin( uTime + TAU / 3.) , .0, 1. );
  float b = clamp(HALF_PI *sin( uTime + 2. * TAU / 3.) , .0, 1. );

  uv *= 2.;
  uv.y = u_resolution.y-uv.y;
  vec2 gv = fract( uv );
  vec4 color = vec4(0.1);
  // color = vec4( smoothstep(vec2(0),u_resolution, uv), 0, 1);
  for ( int i = 0; i < MAX_POSITIONS_LENGTH; i++ ) {
    vec2 position = boids[i].position;
    vec2 not = vec2(-1);
    if ( position.x == not.x && position.y == not.y ) break;
    vec4 boid = PaintBoid( uv, boids[i] );
    color += boid;
  }
  
  // if ( distance( gl_FragCoord.xy, vec2(0) ) < 1. ) color = vec3(1.);
  fragColor = color;
}
`;

class UniformBuffer {
  private buffer: WebGLBuffer;
  constructor(
    private data: Float32Array,
    public bound_location = 0
  ) {

    this.buffer = GL.createBuffer();
    GL.bindBuffer(GL.UNIFORM_BUFFER, this.buffer);
    GL.bufferData(GL.UNIFORM_BUFFER, this.data, GL.DYNAMIC_DRAW);
    GL.bindBuffer(GL.UNIFORM_BUFFER, null);
    GL.bindBufferBase(GL.UNIFORM_BUFFER, this.bound_location, this.buffer);
  }

  update(gl, data, offset = 0) {
    if ( data !== this.data ) this.data.set(data, offset);

    gl.bindBuffer(gl.UNIFORM_BUFFER, this.buffer);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this.data, 0, null);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, this.bound_location, this.buffer);
  }
};
let do_log = true;
export class BoidManager {
  private static program: WebGLProgram;
  private static position: WebGLBuffer;
  private static indices: WebGLBuffer;
  private static boid_buffer: UniformBuffer;
  private static indices_length: number;
  private positions_buffer: Float32Array;
  private a_position: number;
  private u_timestamp: WebGLUniformLocation;
  private u_scale: WebGLUniformLocation;
  private u_translation: WebGLUniformLocation;
  private u_resolution: WebGLUniformLocation;
  private u_rotation: WebGLUniformLocation;

  constructor(
    private boids: Boid[],
    private dimensions: vec4
  ){
    const arr = new Array( 1000 * 4 );
    arr.fill(-1);
    this.positions_buffer = new Float32Array(arr);
    BoidManager.init( this.positions_buffer );
    this.a_position = GL.getAttribLocation(BoidManager.program, 'a_position');
    this.u_timestamp = GL.getUniformLocation(BoidManager.program, 'uTime');
    this.u_scale = GL.getUniformLocation(BoidManager.program, 'u_scale');
    this.u_translation = GL.getUniformLocation(BoidManager.program, 'u_translation');
    this.u_resolution = GL.getUniformLocation(BoidManager.program, 'u_resolution');
    this.u_rotation = GL.getUniformLocation(BoidManager.program, 'u_rotation');
  }

  public static init(positions_buffer : Float32Array): void {
    if ( ! BoidManager.program ) BoidManager.program = init_shader_program( GL, boid_manager_vert_source, boid_manager_frag_source );
    if ( ! BoidManager.position ) {
      const positions = new Float32Array( [
        0, 0,
        GL.canvas.height, 0,
        GL.canvas.height, GL.canvas.width,
        0, GL.canvas.width,
      ] );

      BoidManager.position = GL.createBuffer();
      GL.bindBuffer( GL.ARRAY_BUFFER, BoidManager.position );
      GL.bufferData( GL.ARRAY_BUFFER, positions, GL.STATIC_DRAW );
      GL.bindBuffer( GL.ARRAY_BUFFER, null);
    }
    if ( ! BoidManager.indices ) {
      const indices = new Uint16Array([
        0,1,2,
        0,2,3
      ]);

      BoidManager.indices = GL.createBuffer();
      BoidManager.indices_length = indices.length;
      GL.bindBuffer( GL.ELEMENT_ARRAY_BUFFER, BoidManager.indices );
      GL.bufferData( GL.ELEMENT_ARRAY_BUFFER, indices, GL.STATIC_DRAW );
      GL.bindBuffer( GL.ELEMENT_ARRAY_BUFFER, null);
    }
    if ( ! BoidManager.boid_buffer ) {
      BoidManager.boid_buffer = new UniformBuffer( positions_buffer, 0);
      const program = BoidManager.program;
      GL.uniformBlockBinding( program, GL.getUniformBlockIndex( program, "u_boids" ), BoidManager.boid_buffer.bound_location);
    }
  }

  public flock_and_render( timestamp: number, frametime: number ): QuadTree<Boid> {
    let t;
    const quadtree = new QuadTree(this.boids, this.dimensions, i=>i.position, 10 );
    for ( let boid of this.boids ) {
      const tr = quadtree.query_circle([boid.x, boid.y, 35]);
      if ( ! t ) t = tr;
      boid.hue = tr.length * this.boids.length / Math.E;
      boid.flock( tr, 60 * frametime );
    }
    this.pre_render( timestamp, frametime );
    this.boids.forEach( (boid, i)=>{
      boid.update(60 * frametime);
      this.positions_buffer.set( boid.position, i * 4 );
      this.positions_buffer.set( boid.velocity, i * 4 + 2 );
    });
    BoidManager.boid_buffer.update(GL, this.positions_buffer);
    // if ( Math.floor(frametime % 3000) === 0 ) {
    //   console.log( this.positions_buffer);
    // }
    this.render( timestamp, frametime );
   
    return quadtree;
  }

  private pre_render( timestamp:number, frametime: number ): void {
    GL.useProgram(BoidManager.program);
    // Bind vertex buffer object
    GL.bindBuffer(GL.ARRAY_BUFFER, BoidManager.position);

    // Bind index buffer object
    GL.bindBuffer(GL.ELEMENT_ARRAY_BUFFER, BoidManager.indices);

    // Tell WebGL how to pull out the positions from the position
    // buffer into the vertexPosition attribute.
    {
      const num_components = 2;  // pull out 3 values per iteration
      const type = GL.FLOAT;    // the data in the buffer is 32bit floats
      const normalize = false;  // don't normalize
      const stride = 0;         // how many bytes to get from one set of values to the next
                                // 0 = use type and numComponents above
      const offset = 0;         // how many bytes inside the buffer to start from
      GL.vertexAttribPointer(
          this.a_position,
          num_components,
          type,
          normalize,
          stride,
          offset);
      GL.enableVertexAttribArray(this.a_position);
    }

    // Set the shader uniforms
    GL.uniform2f(this.u_scale,1,1);
    GL.uniform2f(this.u_translation,0,0);
    GL.uniform2f(this.u_resolution, GL.canvas.width, GL.canvas.height);

    // const rotation = vec2.normalize( vec2.create(), this.velocity );

    GL.uniform2f(this.u_rotation, 0,1);

    const uTime = ( timestamp ) / 1000;
    GL.uniform1f(
      this.u_timestamp,
      uTime
    );
    
  }

  private render( timestamp:number, frametime: number ): void {
    {
      const offset = 0;
      const vertexCount = 3;
      GL.drawElements(GL.TRIANGLES, BoidManager.indices_length, GL.UNSIGNED_SHORT, 0);
    }
  }
}
