
#include <device_functions.h>
#include <helper_math.h> 

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <time.h>

#include <timer.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

#define width 1280   //screen width
#define height 720   //screen height

#define MAX_STEP 400
#define MAX_DIST 100.
#define PRECISION 0.01

#define OBJ_IN_SCENE 10

#define N_THREAD 16

#define TopColor            make_float3( 0.35, 0.4, 0.8 )
#define MiddleColor         make_float3( 0.8, 0.8, 0.8 )
#define BottomColor         make_float3( 0.8, 0.4, 0.33 )

#define REFRESH_DELAY     10 //ms

double t = 0.0f;   //timer
float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer

StopWatchInterface* timer = NULL;

float fpsCount = 0; 
float fpsLimit = 1;
double startFrame; 
double endFrame; 
int frameCount = 0; 
float FPS;

//-----------------------
struct RM {
	float dist;
	float travel;
};


struct PARAM
{
	int movingCam; 
	int plane_in_scene;
	int obj_in_scene; 
	int move;
};


struct Blob
{
	int shape;//sphere, cube, tourus
	float3 o_position;
	float3 position;
	float3 color;
	float size;
	int oper; //union, substraction, intersection
	int morph;
	float3 movement;
	int isMoving;
};

__constant__ __device__ Blob blobs_device[10];
Blob blobs_host[10];

__constant__ __device__ PARAM param_device[1];
PARAM param_host[1]; 

//--------------------------------------------------------------- DEVICE
//################ BASIC FUNC
__device__ float mix(float a, float b, float x)
{
	return a * (1 - x) + b * x;
}

__device__ float3 mix(float3 a, float3 b, float x)
{
	float r = mix(a.x, b.y, x);
	float g = mix(a.y, b.y, x);
	float bb = mix(a.z, a.z, x);

	return make_float3(r, g, bb);
}

__device__ float clamp_f(float x, float min_v, float max_v)
{
	return min(max(x, min_v), max_v);
}

__device__ float3 abs(float3 vec)
{
	float3 r;
	r.x = vec.x * ((vec.x < 0) * (-1) + (vec.x > 0));
	r.y = vec.y * ((vec.y < 0) * (-1) + (vec.y > 0));
	r.z = vec.z * ((vec.z < 0) * (-1) + (vec.z > 0));
	return r;
}

//################ OPERATORS
__device__ float smoothUnion(float d1, float d2, float k) {
	float h = clamp_f(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
	return mix(d2, d1, h) - k * h * (1.0 - h);
}

__device__ float smoothSubtraction(float d1, float d2, float k) {
	float h = clamp_f(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
	return mix(d2, -d1, h) + k * h * (1.0 - h);
}

__device__ float changeShape(float dist1, float dist2, float time) //remember this is the k value in raymarching or t in mapping
{
	return mix(dist1, dist2, sin(time) * .5 + .5);
}

//################ DISTANCE FUCTIONS
__device__  float plane(float3 p, float3 c, float3 n)   //plane signed distance field
{
	return dot(p - c, n);
}

__device__ float floor(float3 pos)
{
	return 2 + pos.y;
}

__device__ float sphere(float3 p, float3 sphere_position, float radius)
{
	return length(p - sphere_position) - radius;
}

__device__ float torus(float3 rayPos, float3 pos, float rad) //torus have 2 radius, the main shape and the radius of the border
{
	pos = rayPos - pos;
	float2 radius = make_float2(rad, rad * 0.3);
	float2 q = make_float2(length(make_float2(pos.x, pos.z)) - radius.x, pos.y);
	return length(q) - radius.y;
}

__device__  float tetrahedron(float3 p, float3 pos, float e)   //tetrahedron signed distance field, created from planes intersection
{
	p = pos - p;
	float f = 0.57735;
	float a = plane(p, make_float3(e, e, e), make_float3(-f, f, f));
	float b = plane(p, make_float3(e, -e, -e), make_float3(f, -f, f));
	float c = plane(p, make_float3(-e, e, -e), make_float3(f, f, -f));
	float d = plane(p, make_float3(-e, -e, e), make_float3(-f, -f, -f));
	return max(max(a, b), max(c, d));

}

__device__ float ShapeDistance(float3 pos, Blob blob, float t )
{
	float3 blob_pos = blob.position;// +(make_float3(cos(t * blob.movement.x), cos(t * blob.movement.y), cos(t * blob.movement.z))) * (blob.isMoving * param_device[0].move); //if is moving == 1, else == 0 and there is no add

	if (blob.shape == 0)
		return sphere(pos, blob_pos, blob.size);
	if (blob.shape == 1)
		return torus(pos, blob_pos, blob.size);
	if (blob.shape == 2)
		return tetrahedron(pos, blob_pos, blob.size);
	return 0.0; 
}

__device__ float3 getColor(float3 pos, float time) {
	
	float3 color = (max(0.0, 1.0 - floor(pos)) * make_float3(0.0, 0.4, 0.0) * 1.0 )* param_device[0].plane_in_scene;
	
	for (int i = 0; i < OBJ_IN_SCENE; i++)
		//if (blobs_device[i].oper != 1)
			color += max(0.0, 1.0 - ShapeDistance(pos, blobs_device[i], time)) * blobs_device[i].color * 1.0;

	return color;
}

//################ MAPPING SCENE
__device__ float map(float3 p, float t)   //virtual geometry
{
	float result; 
	result = 1e20;

	if (param_device[0].plane_in_scene == 1)
		result = floor(p);

	//register is more efficient than constant
	int move = param_device[0].movingCam;

	float3 p_c;
	p_c = make_float3(p.x, cos(t) * p.y + sin(t) * p.z, -sin(t) * p.y + cos(t) * p.z);
	p_c = make_float3(cos(t) * p_c.x - sin(t) * p_c.z, p_c.y, sin(t) * p_c.x + cos(t) * p_c.z);
	//    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//insolita istruzione per rimuovere un if (move == 0) p = p_c 
	//sommo a p la posizione p_c - p (perchè non voglio davvero sommare p_c a p) e moltiplico il risultato per move
	//move è uno se c'è movimento, quindi  p = p + p_c - p * 1 --> p =  p_c
	//move è zero se non c'è movimento  p = p + (p_c - p) * 0 --> p = p
	p = p + (p_c-p) * move ; 
	
	//for all shapes
	//for(int i = 0; i < param_device[0].obj_in_scene; i++)
		//shapes[i] = blobs_device[i].morph == 0 ? ShapeDistance(p, blobs[i]) : ChangeShape(blobs[i], p);

	//For all unions operator
	for (int i = 0; i < param_device[0].obj_in_scene; i++) {
		if (blobs_device[i].oper == 0) //is union operator
			result = smoothUnion(result, ShapeDistance(p, blobs_device[i], t), 0.5);
	}
/*
	for (int i = 0; i < OBJ_IN_SCENE; i++)
		if (blobs_device[i].oper == 1) //is union operator
			result = smoothSubtraction(ShapeDistance(p, blobs_device[i], t), result, 0.5);
*/	
	return result; 
}

//################ RAYMARCH
__device__ RM raymarch(float3 ro, float3 rd, float time)   //raymarching
{
	float travel = 0.0;
	float hit; 

	for (int i = 0; i < MAX_STEP; i++)
	{
		float3 point = ro + travel * rd;
		hit = map(point, time);
		travel += hit;

		if (hit < PRECISION || travel > MAX_DIST) break;
	}

	RM result; 
	result.travel = travel; 
	//result.dist = hit; 

	return result; 
}

//################ RENDERING POINT

__device__ float3 GetNormal(float3 point, float t)
{
	float base = map(point, t);

	float2 epsilon = make_float2(0.01, 0.0);

	float3 normal = base - make_float3(
		map(point - make_float3(0.01,0.0,0.0),t), //per capire lo slope, comparo i punti vicini al punto su cui calcolare la norm
		map(point - make_float3(0.0, 0.01, 0.0),t),
		map(point - make_float3(0.0, 0.0, 0.01),t));

	return normalize(normal);
}

__device__ float3 render_point(float3 ro, float3 p, float t, float3 color)   //directional derivative based lighting
{
	float3 lightPosition = make_float3(0.0,5.0,-2.0);
	float2 movLight = make_float2(sin(t * 0.5) * 4., cos(t * 0.5) * 4.0);
	lightPosition += make_float3(movLight, 0.0);

	float3 light = normalize(lightPosition - p);

	float3 normal = GetNormal(p, t);
	
	float3 finalColor = normal;
	float3 toCamera = normalize(ro - p);

	float shadowHit;
	bool shadow;
	//Shadow color	
	shadowHit = raymarch(p + (normal * PRECISION * 2.), light, t).travel;
	shadow = shadowHit < length(p - lightPosition);
	//_synchthreads() <- not here, because i have render_bg that could cause deadlock
	//have to syncr the block because the raymarching may misalign threads

	float diffuse = clamp(dot(normal, light), 0.0, 1.0); //faccio il clamp in modo da non aver un valore negativo
	float3 diffuseColor = diffuse * color;
	float specular = diffuse;
	float3 specularColor = diffuseColor;

	if (!shadow)
	{
		float3 reflectedLight = normalize(reflect(-light, normal));
		specular = pow((double)clamp(dot(reflectedLight, light), 0.0, 1.0), 5.0);

		specular = min(diffuse, specular);
		
		specularColor = specular * make_float3(1.0, 1.0, 1.0); //specular color 1,1,1
		
		finalColor = clamp((diffuseColor + specularColor),0.0,1.0);
	}
	else finalColor = float3(diffuseColor) * 0.4;
	
	return finalColor;
}

__device__ float3 render_bg(float2 uv)
{
	
	float3 color = make_float3(0.0, 0.0,0.0);
	
	if (uv.y > 0.0)	color = mix(MiddleColor, TopColor, uv.y*2);

	if (uv.y <= 0.0) color = mix(MiddleColor, BottomColor, uv.y * -2);
	
	//color = make_float3(uv.x, 0.0, uv.y);
	return color;
}


//################ RAYMARCHING MAIN
//++++++++++++++++++++ 3
__global__ void rendering(float3* output, float k)
{
	//get coordinate of pixel
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;
	
	//if(x==0 && y==0)
		//printf("\n\n[%d]-> isMoving: %d,     color: %1.4f - %1.4f - %1.4f,     position: %2.4f, %2.4f, %2.4f", 5, blobs_device[5].isMoving, blobs_device[5].color.x, blobs_device[5].color.y, blobs_device[5].color.z, blobs_device[5].position.x, blobs_device[5].position.y, blobs_device[5].position.z);

	float2 resolution = make_float2((float)width, (float)height);   //screen resolution
	float2 coordinates = make_float2((float)x, (float)y);   //fragment coordinates
	
	//float2 uv = (2.0 * coordinates - resolution) / resolution.y;
	float2 uv = coordinates / resolution;
	uv -= 0.5;
	uv.x *= resolution.x / resolution.y;

	float3 ro = make_float3(0.0f, 0.0f, -20.0f);   //ray origin
	float3 rd = normalize(make_float3(uv, 1.0f));   //ray direction
	RM raym = raymarch(ro, rd, k);

	//_synchthreads();
	float dist = raym.travel;
	
	float3 point = ro + dist * rd;

	float3 c;

	if (dist > MAX_DIST) c = render_bg(uv);
	else c = render_point(ro, point, k, 1(point, k));
	
	//else c = make_float3(dist, dist, dist);

	float colour;
	unsigned char bytes[] = { (unsigned char)(c.x * 255 + 0.5), (unsigned char)(c.y * 255 + 0.5), (unsigned char)(c.z * 255 + 0.5), 1 };
	memcpy(&colour, &bytes, sizeof(colour));   //convert from 4 bytes to single float
	output[i] = make_float3(x, y, colour);
}

//#################################### TIMER FUNCTIONS
void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		
		float ifps =  1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
		sprintf(fps, "fps: %3.f fps ", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.0f);

		sdkResetTimer(&timer);
	}
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
		t += 0.01667777f;
	}
}

//############################### DISPLAY LOOP
//++++++++++++++++++++ 2
void display(void)
{
	sdkStartTimer(&timer); 
	
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)& device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);
	
	dim3 block(N_THREAD, N_THREAD, 1);
	dim3 grid(width / block.x, height / block.y, 1);


	rendering << < grid, block >> > (device, t);   //execute kernel
	cudaThreadSynchronize();
	
	cudaGLUnmapBufferObject(buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glutSwapBuffers();

	sdkStopTimer(&timer);

	computeFPS(); 
}

//########################################################################################################### HOST
void changeSize(int w, int h)
{
	//std::cout << "w " << w << " h " << h << std::endl;
	glutReshapeWindow(width, height);
}

void keyboard(unsigned char key, int x, int y)
{
	int i;
	int* j = &i;
	switch (key){

		case 'f':
			if (param_host[0].plane_in_scene == 0) {
				std::cout << "add plane in scene" << std::endl;
				param_host[0].plane_in_scene = 1;
				cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			}
			else {
				std::cout << "remove plane from scene" << std::endl;
				param_host[0].plane_in_scene = 0;
				cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			}break;

		case '+':
			std::cout << "add object in scene (MAX 10 OBJ)" << std::endl;
			param_host[0].obj_in_scene = param_host[0].obj_in_scene < 10 ? param_host[0].obj_in_scene +1 : param_host[0].obj_in_scene;
			cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			break;

		case '-':
			std::cout << "remove object in scene (not negative)" << std::endl;
			param_host[0].obj_in_scene = param_host[0].obj_in_scene > 0 ? param_host[0].obj_in_scene - 1 : param_host[0].obj_in_scene;
			cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			break;

		case 'm':
			std::cout << "move camera" << std::endl;
			param_host[0].movingCam = param_host[0].movingCam == 0 ? 1 : 0;
			cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			break;

		case 'a':
			std::cout << "animate obj" << std::endl;
			param_host[0].move = param_host[0].move == 0 ? 1 : 0;
			cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);
			break;

		case 'h':
			std::cout << "\na: animate obj \nf: add floor or remove floor \n+: add object \n-:remove obj\n(obj in scene: " << param_host[0].obj_in_scene <<") \nm: move camera \nh: print help" << std::endl;
			break;


	}
	glutPostRedisplay();
}

//++++++++++++++++++++ 1
int main(int argc, char** argv)
{
	glutInit(&argc, argv);   //OpenGL initializing
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	
	//creating window
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("Basic CUDA OpenGL raymarching - tryy");

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	glutDisplayFunc(display); //register the call back 

	sdkCreateTimer(&timer);

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(keyboard);

	glewInit();
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	unsigned int size = width * height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
	
	srand(time(NULL));
	for (int i = 0; i < OBJ_IN_SCENE; i++)
	{
		//SETTING SCENE of 10 obj with random properties
		float size = i;
		float x = size;
		float y = x / 2;
		float z = y / 2;

		x = int(x) % 2 == 0 ? 3.0 : -3.0;
		y = (int)y % 2 == 0 ? 3.0 : -3.0;
		z = (int)z % 2 == 0 ? 3.0 : -3.0;

		if (size > 7) {
			x = z = 0.0;
			y = (int)size % 2;
		}
		Blob newObject;
		float k = ((float)rand() / RAND_MAX) + 0.50;
		//---------Pos
		newObject.position = make_float3(x + k, y + k, z + k);
		//---------Size
		newObject.size = k;
		//---------Color
		newObject.color = make_float3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		//---------Shape
		newObject.shape = rand() % 3; //0 sphere, 1 torus, 2 tetrahedreon
		//---------Movement
		newObject.movement = make_float3(rand() % 5, rand() % 5, rand() % 5);
		newObject.isMoving = rand() % 2;
		//---------Oper
		newObject.oper = 0;//rand() % 2; //0 union, 1 subtraction
		//---------Morph
		newObject.morph = rand() % 2; //1 true, 0 false

		blobs_host[i] = newObject;
	}
	
	for (int i = 0; i < OBJ_IN_SCENE; i++)
		printf("\n[%d]: Shape: %d | position: %2.4f - %2.4f - %2.4f\t| color: %1.4f - %1.4f - %1.4f | size: %1.4f | oper: %d | morph: %d | isMoving: %d | movment: %2.4f - %2.4f - %2.4f ", i, blobs_host[i].shape, blobs_host[i].position.x, blobs_host[i].position.y, blobs_host[i].position.z, blobs_host[i].color.x, blobs_host[i].color.y, blobs_host[i].color.z, blobs_host[i].size, blobs_host[i].oper, blobs_host[i].morph, blobs_host[i].isMoving, blobs_host[i].movement.x, blobs_host[i].movement.y, blobs_host[i].movement.z);

	param_host[0].movingCam = 0; 
	param_host[0].obj_in_scene = 10; 
	param_host[0].plane_in_scene = 0; 
	param_host[0].move = 0; 

	cudaMemcpyToSymbol(param_device, param_host, sizeof(struct PARAM), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(blobs_device, blobs_host, sizeof(struct Blob) * 10, 0, cudaMemcpyHostToDevice);
	cudaMalloc(&device, width * height * sizeof(float3));   //allocate memory on the GPU VRAM


	std::cout << "\na: animate obj \nf: add floor or remove floor \n+: add object \n-:remove obj\n(obj in scene: " << param_host[0].obj_in_scene << ") \nm: move camera \nh: print help" << std::endl;

	glutMainLoop();   //event processing loop
	cudaFree(device);
}