#include <cstdlib>

#include <iostream>
#include <string>

#define GLEW_STATIC // Specify GLEW_STATIC to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <glut.h>

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "Postprocess.h"

#define SRC_BUFFER  0
#define DST_BUFFER  1

int g_iGLUTWindowHandle = 0;
int g_iWindowPositionX = 0;
int g_iWindowPositionY = 0;
int g_iWindowWidth = 512;
int g_iWindowHeight = 512;

int g_iImageWidth = g_iWindowWidth;
int g_iImageHeight = g_iWindowHeight;

float g_fRotate[3] = { 0.0f, 0.0f, 0.0f };  // Rotation parameter for scene object.
bool g_bAnimate = true;                     // Animate the scene object
bool g_bPostProcess = true;                 // Enable/disable the postprocess effect.
float g_fBlurRadius = 2.0f;                 // Radius of 2D convolution blur performed in postprocess step.

GLuint g_GLFramebuffer = 0;                  // Frame buffer object for off-screen rendering.
GLuint g_GLColorAttachment0 = 0;            // Color texture to attach to frame buffer object.
GLuint g_GLDepthAttachment = 0;             // Depth buffer to attach to frame buffer object.

GLuint g_GLPostprocessTexture = 0;          // This is where the result of the post-process effect will go.
                                            // This is also the final texture that will be blit to the back buffer for viewing.
                                            
// The CUDA Graphics Resource is used to map the OpenGL texture to a CUDA
// buffer that can be used in a CUDA kernel.
// We need 2 resource: One will be used to map to the color attachment of the
//   framebuffer and used read-only from the CUDA kernel (SRC_BUFFER), 
//   the second is used to write the postprocess effect to (DST_BUFFER).
cudaGraphicsResource_t g_CUDAGraphicsResource[2] = { 0, 0 };   

// Initialize OpenGL/GLUT
bool InitGL( int argc, char* argv[] );
// Initialize CUDA for OpenGL
void InitCUDA();
// Render a texture object to the current framebuffer
void DisplayImage( GLuint texture, unsigned int x, unsigned int y, unsigned int width, unsigned int height );


// Create a framebuffer object that is used for offscreen rendering.
void CreateFramebuffer( GLuint& framebuffer, GLuint colorAttachment0, GLuint depthAttachment );
void DeleteFramebuffer( GLuint& framebuffer );

void CreatePBO( GLuint& bufferID, size_t size );
void DeletePBO( GLuint& bufferID );

void CreateTexture( GLuint& texture, unsigned int width, unsigned int height );
void DeleteTexture( GLuint& texture );

void CreateDepthBuffer( GLuint& depthBuffer, unsigned int width, unsigned int height );
void DeleteDepthBuffer( GLuint& depthBuffer );

// Links a OpenGL texture object to a CUDA resource that can be used in the CUDA kernel.
void CreateCUDAResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags );
void DeleteCUDAResource( cudaGraphicsResource_t& cudaResource );

void IdleGL();
void DisplayGL();
void KeyboardGL( unsigned char key, int x, int y );
void ReshapeGL( int w, int h );

// Define a few default filters to apply to the image
// Filter definitions found on http://ian-albert.com/old/custom_filters/

#define SCALE_INDEX 25  // The index of the scale value in the filter
#define OFFSET_INDEX 26 // The index of the offset value in the filter

// Filter matrices are defined by a series of 25 values which represent the weights of the 
// each neighboring pixel followed by the default scale and the offset that is applied to each pixel.
float g_Unfiltered[] = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0,
    1, 0
};

float g_BlurFilter[] = { 
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 1, 
    1, 2, 3, 2, 1,
    1, 2, 2, 2, 2,
    1, 1, 1, 1, 1,
    35, 0
};

float g_SharpeningFilter[] = {
    0,  0,  0,  0,  0,
    0,  0, -1,  0,  0, 
    0, -1,  5, -1,  0,
    0,  0, -1,  0,  0, 
    0,  0,  0,  0,  0,
    1, 0
};

float g_EmbossFilter[] = { 
    0, 0, 0,  0, 0, 
    0, 0, 0,  0, 0,
    0, 0, 1,  0, 0,
    0, 0, 0, -1, 0,
    0, 0, 0,  0, 0,
    1, 128
};

float g_InvertFilter[] = {
    0, 0,  0, 0, 0,
    0, 0,  0, 0, 0, 
    0, 0, -1, 0, 0,
    0, 0,  0, 0, 0,
    0, 0,  0, 0, 0,
    1, 255
};

float g_EdgeFilter[] = {
    0,  0,  0,  0,  0,
    0, -1, -1, -1,  0, 
    0, -1,  8, -1,  0,
    0, -1, -1, -1,  0, 
    0,  0,  0,  0,  0,
    1, 0
};

// The current scale
float g_Scale = 1.0f;
// The current offset
float g_Offset = 0.0f;
// The currently selected filter
float* g_CurrentFilter;

void Cleanup( int errorCode, bool bExit = true )
{
    if ( g_iGLUTWindowHandle != 0 )
    {
        glutDestroyWindow( g_iGLUTWindowHandle );
        g_iGLUTWindowHandle = 0;
    }
    if ( bExit )
    {
        exit( errorCode );
    }
}

// Create a pixel buffer object
void CreatePBO( GLuint& bufferID, size_t size )
{
    // Make sure the buffer doesn't already exist
    DeletePBO( bufferID );

    glGenBuffers( 1, &bufferID );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_STREAM_DRAW );

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
}

void DeletePBO(  GLuint& bufferID )
{
    if ( bufferID != 0 )
    {
        glDeleteBuffers( 1, &bufferID );
        bufferID = 0;
    }
}

// Create a texture resource for rendering to.
void CreateTexture( GLuint& texture, unsigned int width, unsigned int height )
{
    // Make sure we don't already have a texture defined here
    DeleteTexture( texture );

    glGenTextures( 1, &texture );
    glBindTexture( GL_TEXTURE_2D, texture );

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create texture data (4-component unsigned byte)
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );

    // Unbind the texture
    glBindTexture( GL_TEXTURE_2D, 0 );
}

void DeleteTexture( GLuint& texture )
{
    if ( texture != 0 )
    {
        glDeleteTextures(1, &texture );
        texture = 0;
    }
}

void CreateDepthBuffer( GLuint& depthBuffer, unsigned int width, unsigned int height )
{
    // Delete the existing depth buffer if there is one.
    DeleteDepthBuffer( depthBuffer );

    glGenRenderbuffers( 1, &depthBuffer );
    glBindRenderbuffer( GL_RENDERBUFFER, depthBuffer );

    glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );

    // Unbind the depth buffer
    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

void DeleteDepthBuffer( GLuint& depthBuffer )
{
    if ( depthBuffer != 0 )
    {
        glDeleteRenderbuffers( 1, &depthBuffer );
        depthBuffer = 0;
    }
}

void CreateFramebuffer( GLuint& framebuffer, GLuint colorAttachment0, GLuint depthAttachment )
{
    // Delete the existing framebuffer if it exists.
    DeleteFramebuffer( framebuffer );

    glGenFramebuffers( 1, &framebuffer );
    glBindFramebuffer( GL_FRAMEBUFFER, framebuffer );

    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachment0, 0 );
    glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthAttachment );

    // Check to see if the frame buffer is valid
    GLenum fboStatus = glCheckFramebufferStatus( GL_FRAMEBUFFER );
    if ( fboStatus != GL_FRAMEBUFFER_COMPLETE )
    {
        std::cerr << "ERROR: Incomplete framebuffer status." << std::endl;
    }

    // Unbind the frame buffer
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

void DeleteFramebuffer( GLuint& framebuffer )
{
    if ( framebuffer != 0 )
    {
        glDeleteFramebuffers( 1, &framebuffer );
        framebuffer = 0;
    }
}

void CreateCUDAResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags )
{
    // Map the GL texture resource with the CUDA resource
    cudaGraphicsGLRegisterImage( &cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags );
}

void DeleteCUDAResource( cudaGraphicsResource_t& cudaResource )
{
    if ( cudaResource != 0 )
    {
        cudaGraphicsUnregisterResource( cudaResource );
        cudaResource = 0;
    }
}

int main( int argc, char* argv[] )
{
    std::cout << "===============================================================================" << std::endl;
    std::cout << "Press [space] to toggle the rotation of the teapot." << std::endl;
    std::cout << "Press [enter] to toggle the post-process effect." << std::endl;
    std::cout << "Press [1] to show an unfiltered effect." << std::endl;
    std::cout << "Press [2] to show a blurred effect." << std::endl;
    std::cout << "Press [3] to show a sharpen effect." << std::endl;
    std::cout << "Press [4] to show an emboss effect." << std::endl;
    std::cout << "Press [5] to show an invert effect." << std::endl;
    std::cout << "Press [6] to show an edge-detect effect." << std::endl;
    std::cout << "Press [esc] to end the program." << std::endl;
    std::cout << "===============================================================================" << std::endl;

    // Init GLUT
    if ( !InitGL( argc, argv ) )
    {
        std::cerr << "ERROR: Failed to initialize OpenGL" << std::endl;
    }

    InitCUDA();

    // By default, no filter will be applied.
    g_CurrentFilter = g_Unfiltered;
    g_Scale = g_CurrentFilter[SCALE_INDEX];
    g_Offset = g_CurrentFilter[OFFSET_INDEX];

    // Startup our GL render loop
    glutMainLoop();

}

bool InitGL( int argc, char* argv[] )
{
    // Material property constants.
    const GLfloat fRed[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    const GLfloat fWhite[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    int iScreenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int iScreenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowPosition( iScreenWidth / 2 - g_iWindowWidth / 2,
        iScreenHeight / 2 - g_iWindowHeight / 2 );
    glutInitWindowSize( g_iWindowWidth, g_iWindowHeight );

    g_iGLUTWindowHandle = glutCreateWindow( "Postprocess GL" );

    // Register GLUT callbacks
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutReshapeFunc(ReshapeGL);
    glutIdleFunc(IdleGL);

    // Init GLEW
    glewInit();
    GLboolean gGLEW = glewIsSupported(
        "GL_VERSION_3_1 " 
        "GL_ARB_pixel_buffer_object "
        "GL_ARB_framebuffer_object "
        "GL_ARB_copy_buffer " 
        );

    int maxAttachemnts = 0;
    glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS, &maxAttachemnts );

    if ( !gGLEW ) return false;

    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );
    glDisable( GL_DEPTH_TEST );

    // Setup the viewport
    glViewport( 0, 0, g_iWindowWidth, g_iWindowHeight );

    // Setup the projection matrix
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    gluPerspective( 60.0, (GLdouble)g_iWindowWidth/(GLdouble)g_iWindowHeight, 0.1, 1.0 );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    // Enable one light.
    glEnable( GL_LIGHT0 );
    glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, fRed );
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, fWhite );
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 60.0f );

    return true;
}

void InitCUDA()
{
    // We have to call cudaGLSetGLDevice if we want to use OpenGL interoperability.
    cudaGLSetGLDevice(0);
}


void IdleGL()
{
    if (g_bAnimate) 
    {
        g_fRotate[0] += 0.2; while(g_fRotate[0] > 360.0f) g_fRotate[0] -= 360.0f;   // Increment and clamp
        g_fRotate[1] += 0.6; while(g_fRotate[1] > 360.0f) g_fRotate[1] -= 360.0f;
        g_fRotate[2] += 1.0; while(g_fRotate[2] > 360.0f) g_fRotate[2] -= 360.0f;
    }

    glutPostRedisplay();

}

// Render the initial scene
void RenderScene()
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( 60.0, (GLdouble)g_iWindowWidth / (GLdouble)g_iWindowHeight, 0.1, 10.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( 0.0f, 0.0f, -3.0f );

    glRotatef( g_fRotate[0], 1.0f, 0.0f, 0.0f );
    glRotatef( g_fRotate[1], 0.0f, 1.0f, 0.0f );
    glRotatef( g_fRotate[2], 0.0f, 0.0f, 1.0f );

    glViewport( 0, 0, g_iWindowWidth, g_iWindowHeight );

    glEnable( GL_LIGHTING );
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LESS );

    glutSolidTeapot( 1.0 );

}

// Perform a post-process effect on the current framebuffer (back buffer)
void Postprocess()
{
    if ( g_bPostProcess )
    {
        PostprocessCUDA( g_CUDAGraphicsResource[DST_BUFFER], g_CUDAGraphicsResource[SRC_BUFFER], g_iImageWidth, g_iImageHeight, g_CurrentFilter, g_Scale, g_Offset );
    }
    else
    {
        // No postprocess effect. Just copy the contents of the color buffer
        // from the framebuffer (where the scene was rendered) to the 
        // post-process texture.  The postprocess texture will be rendered to the screen
        // in the next step.
        glBindFramebuffer( GL_FRAMEBUFFER, g_GLFramebuffer );
        glBindTexture( GL_TEXTURE_2D, g_GLPostprocessTexture );

        glCopyTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, g_iImageWidth, g_iImageHeight, 0 );

        glBindTexture( GL_TEXTURE_2D, 0 );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    }
}

void DisplayGL()
{
    // Bind the framebuffer that we want to use as the render target.
    glBindFramebuffer( GL_FRAMEBUFFER, g_GLFramebuffer );
    RenderScene();
    // Unbind the framebuffer so we render to the back buffer again.
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    Postprocess();

    // Blit the image full-screen
    DisplayImage( g_GLPostprocessTexture, 0, 0, g_iWindowWidth, g_iWindowHeight );

    glutSwapBuffers();
    glutPostRedisplay();
}

void DisplayImage( GLuint texture, unsigned int x, unsigned int y, unsigned int width, unsigned int height )
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glPushAttrib( GL_VIEWPORT_BIT );
    glViewport(x, y, width, height );

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glPopAttrib();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

void KeyboardGL( unsigned char key, int x, int y )
{
    switch( key )
    {
    case '1':
        {
            std::cout << "Filter: Unfiltered" << std::endl;
            g_CurrentFilter = g_Unfiltered;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];
        }
        break;
    case '2':
        {
            std::cout << "Filter: Blur" << std::endl;
            g_CurrentFilter = g_BlurFilter;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];
        }
        break;
    case '3':
        {
            std::cout << "Filter: Sharpening" << std::endl;
            g_CurrentFilter = g_SharpeningFilter;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];
        }
        break;
    case '4':
        {
            std::cout << "Filter: Emboss" << std::endl;
            g_CurrentFilter = g_EmbossFilter;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];

        }
        break;
    case '5':
        {
            std::cout << "Filter: Invert" << std::endl;
            g_CurrentFilter = g_InvertFilter;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];
        }
        break;
    case '6':
        {
            std::cout << "Filter: Edge Detection" << std::endl;
            g_CurrentFilter = g_EdgeFilter;
            g_Scale = g_CurrentFilter[SCALE_INDEX];
            g_Offset = g_CurrentFilter[OFFSET_INDEX];
        }
        break;
    case ' ':
        {
            // Toggle animation
            g_bAnimate = !g_bAnimate;
        }
        break;
    case '\012': // return key
    case '\015': // enter key
        {
            g_bPostProcess = !g_bPostProcess;
            std::cout << "Postprocess effect: ";
            std::cout << ( g_bPostProcess ? "ON" : "OFF" ) << std::endl;
        }
        break;
    case '\033': // escape quits
    case 'Q':    // Q quits
    case 'q':    // q quits
        {
            // Cleanup up and quit
            Cleanup(0);
        }
        break;
    }

    glutPostRedisplay();
}

void ReshapeGL( int w, int h )
{
    h = std::max(h, 1);

    g_iWindowWidth = w;
    g_iWindowHeight = h;

    g_iImageWidth = w;
    g_iImageHeight = h;

    // Create a surface texture to render the scene to.
    CreateTexture( g_GLColorAttachment0, g_iImageWidth, g_iImageHeight );
    // Create a depth buffer for the frame buffer object.
    CreateDepthBuffer( g_GLDepthAttachment, g_iImageWidth, g_iImageHeight );
    // Attach the color and depth textures to the framebuffer.
    CreateFramebuffer( g_GLFramebuffer, g_GLColorAttachment0, g_GLDepthAttachment );

    // Create a texture to render the post-process effect to.
    CreateTexture( g_GLPostprocessTexture, g_iImageWidth, g_iImageHeight );

    // Map the color attachment to a CUDA graphics resource so we can read it in a CUDA a kernel.
    CreateCUDAResource( g_CUDAGraphicsResource[SRC_BUFFER], g_GLColorAttachment0, cudaGraphicsMapFlagsReadOnly );
    // Map the post-process texture to the CUDA resource so it can be 
    // written in the kernel.
    CreateCUDAResource( g_CUDAGraphicsResource[DST_BUFFER], g_GLPostprocessTexture, cudaGraphicsMapFlagsWriteDiscard );

    glutPostRedisplay();
}
