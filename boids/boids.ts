import {mat4, vec2} from 'gl-matrix';


let GL: WebGL2RenderingContext;
const boid_vert_source = `
precision mediump float;
uniform float uTime;

attribute vec2 a_position;

uniform vec2 u_resolution;
uniform vec2 u_translation;
uniform vec2 u_rotation;
uniform vec2 u_scale;

void main() {
  // Scale the position
  vec2 scaledPosition = a_position * u_scale;

  // Rotate the position
  vec2 rotatedPosition = vec2(
     scaledPosition.x * u_rotation.y + scaledPosition.y * u_rotation.x,
     scaledPosition.y * u_rotation.y - scaledPosition.x * u_rotation.x);

  // Add in the translation.
  vec2 position = rotatedPosition + u_translation;

  // convert the position from pixels to 0.0 to 1.0
  vec2 zeroToOne = position / u_resolution;

  // convert from 0->1 to 0->2
  vec2 zeroToTwo = zeroToOne * 2.0;

  // convert from 0->2 to -1->+1 (clipspace)
  vec2 clipSpace = zeroToTwo - 1.0;

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
    throw 'An error occurred compiling the shaders: ' + info;
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

export class Boid {
  private static program: WebGLProgram;
  private a_position: number;
  private u_timestamp: WebGLUniformLocation;
  private u_scale: WebGLUniformLocation;
  private u_translation: WebGLUniformLocation;
  private u_resolution: WebGLUniformLocation;
  private u_rotation: WebGLUniformLocation;
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