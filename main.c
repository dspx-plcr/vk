#include <err.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "linmath.h"

#define ARR_SZ(a) (sizeof(a) / sizeof((a)[0]))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define PI ((double)3.14159265358979323846)

static struct {
	GLFWwindow *window;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	VkSemaphore *semaphores;
	size_t nsems;
	VkImage *images;
	size_t nims;
	VkImageView *views;
	size_t nviews;
	VkFramebuffer *framebuffers;
	size_t nfbs;

	VkSurfaceCapabilitiesKHR caps;
	VkSurfaceFormatKHR fmt;
	VkPresentModeKHR mode;
} display;

static struct {
	VkDescriptorSetLayout dsetlayout;
	VkPipelineLayout layout;
	VkPipeline pipeline;
	VkShaderModule vertshader;
	VkShaderModule fragshader;
	VkDeviceMemory memory;
	VkBuffer vertbuf;
	VkBuffer transbuf;
	VkBuffer modelbuf;
} modelpipeline;

static struct {
	VkRenderPass pass;
	VkDescriptorPool dpool;
	VkCommandPool cpool;
	VkFence fence;
	VkSemaphore acquiresem;
	VkCommandBuffer cmdbuf;
	VkDeviceMemory memory;

	VkImage modeldepth;
	VkImageView mdview;
} renderer;

static const char *progname = "My Little Vulkan App";
static const float width = 1920;
static const float height = 1080;
static const float depth = 1000;
static const float meshpct = 0.9;

static const double panspeed = 0.2;
static const double rotspeed = PI / (2 * width);
static const double zoomspeed = 70;

static VkInstance instance;
static VkDevice device;
static uint32_t qfamidx;
static uint32_t memidx;

struct boundingbox {
	vec3 offset;
	vec3 extent;
	vec3 dir; 
};

struct bounds {
	struct boundingbox orig;
	struct boundingbox curr;
};

struct object {
	size_t nprims;
	size_t meshidx;
};

struct vertexinfo {
	vec3 pos;
	vec3 norm;
};

static size_t nobjs;
static struct object *objs;
static struct vertexinfo *meshes;
static size_t meshcnt;
static size_t meshsz;
static struct bounds *bounds;
static vec3 camera;
static vec3 target;
static vec3 up;
static mat4x4 *transforms;
static mat4x4 *models;

static vec2 lastmouse;
static bool panning;
static bool rotating;

static int64_t
nanotimerdiff(struct timespec a, struct timespec b)
{
	return 1e9*(a.tv_sec-b.tv_sec) + (a.tv_nsec-b.tv_nsec);
}

static void
reset_camera(void)
{
	target[0] = 0;
	target[1] = 0;
	target[2] = 0;
	up[0] = 0;
	up[1] = -1;
	up[2] = 0;
	camera[0] = 0;
	camera[1] = height/4;
	camera[2] = depth/2;
}

static const char *
vkstrerror(VkResult res)
{
	switch (res) {
	case VK_SUCCESS: return "success";
	case VK_NOT_READY: return "not ready";
	case VK_TIMEOUT: return "timeout";
	case VK_EVENT_SET: return "event set";
	case VK_EVENT_RESET: return "event reset";
	case VK_INCOMPLETE: return "incomplete";
	case VK_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
	case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
	case VK_ERROR_INITIALIZATION_FAILED: return "initialisation failed";
	case VK_ERROR_DEVICE_LOST: return "device lost";
	case VK_ERROR_MEMORY_MAP_FAILED: return "memory map failed";
	case VK_ERROR_LAYER_NOT_PRESENT: return "layer not present";
	case VK_ERROR_EXTENSION_NOT_PRESENT: return "extension not present";
	case VK_ERROR_FEATURE_NOT_PRESENT: return "feature not present";
	case VK_ERROR_INCOMPATIBLE_DRIVER: return "incompatible driver";
	case VK_ERROR_TOO_MANY_OBJECTS: return "too many object";
	case VK_ERROR_FORMAT_NOT_SUPPORTED: return "format not supported";
	case VK_ERROR_FRAGMENTED_POOL: return "fragmented pool";
	case VK_ERROR_UNKNOWN: return "unkown error";
	case VK_ERROR_SURFACE_LOST_KHR: return "surface lost";
	case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "native window in use";
	default: return "<bad error code>";
	}
}

static bool
usabledev(
	VkPhysicalDevice dev,
	VkPhysicalDeviceProperties2 props,
	uint32_t *outqfamidx,
	uint32_t *outmemidx
) {
	if (VK_API_VERSION_MAJOR(props.properties.apiVersion) != 1 ||
			VK_API_VERSION_MINOR(props.properties.apiVersion) < 1)
		return false;

	if (props.properties.limits.maxViewportDimensions[0] < width ||
			props.properties.limits.maxViewportDimensions[1] < height)
		return false;

	uint32_t qcnt;
	vkGetPhysicalDeviceQueueFamilyProperties2(dev, &qcnt, NULL);
	if (SIZE_MAX / sizeof(VkQueueFamilyProperties2) < (size_t)qcnt ||
			(size_t)qcnt != qcnt) {
		warnx("can't allocate space for %zu queue family properties"
			" (%zu * sizeof(VkQueueFamilyProperties2) > SIZE_MAX)",
			qcnt, qcnt);
		return false;
	}
	VkQueueFamilyProperties2 *qprops =
		malloc(sizeof(VkQueueFamilyProperties2) * qcnt);
	if (qprops == NULL)
		err(1, "couldn't allocate for queue family properties");
	for (size_t i = 0; i < qcnt; i++)
		qprops[i] = (VkQueueFamilyProperties2){
			.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
			.pNext = NULL,
		};
	vkGetPhysicalDeviceQueueFamilyProperties2(dev, &qcnt, qprops);
	bool found = false;
	for (size_t i = 0; i < qcnt; i++) {
		VkQueueFlags flags = qprops[i].queueFamilyProperties.queueFlags;
		if (!(flags & VK_QUEUE_GRAPHICS_BIT))
			continue;
		if (glfwGetPhysicalDevicePresentationSupport(
				instance, dev, i)) {
			found = true;
			*outqfamidx = i;
			break;
		}
	}
	free(qprops);
	if (!found)
		return false;

	VkPhysicalDeviceMemoryProperties2 memprops = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		.pNext = NULL,
	};
	vkGetPhysicalDeviceMemoryProperties2(dev, &memprops);
	VkPhysicalDeviceMemoryProperties mp = memprops.memoryProperties;
	found = false;
	for (size_t i = 0; i < mp.memoryTypeCount; i++) {
		VkMemoryPropertyFlags flags = mp.memoryTypes[i].propertyFlags;
		if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
				(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
			*outmemidx = i;
			found = true;
			break;
		}
	}

	return found;
}

static bool
preferdev(VkPhysicalDeviceProperties2 new, VkPhysicalDeviceProperties2 old)
{
	if (old.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
		return true;
	if (new.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			old.properties.deviceType !=
			VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		return true;

	return false;
}

enum displaystage {
	INSTANCE,
	SURFACE,
	DEVICE,
	SWAP_CHAIN,
	SEMAPHORES,
	IMAGE_VIEWS,
	DISPLAY_ALL,
};

static void
cleanup_display(enum displaystage stage)
{
	switch (stage) {
	case DISPLAY_ALL:
	case IMAGE_VIEWS:
		for (size_t i = 0; i < display.nviews; i++)
			vkDestroyImageView(device, display.views[i],
				NULL /* allocator */);
		display.nviews = 0;
	case SEMAPHORES:
		for (size_t i = 0; i < display.nsems; i++)
			vkDestroySemaphore(device, display.semaphores[i],
				NULL /* allocator */);
		display.nsems = 0;
	case SWAP_CHAIN:
		vkDestroySwapchainKHR(
			device, display.swapchain, NULL /* allocator */);
	case DEVICE:
		vkDestroyDevice(device, NULL /* allocator */);
	case SURFACE:
		vkDestroySurfaceKHR(
			instance, display.surface, NULL /* allocator */);
	case INSTANCE:
		vkDestroyInstance(instance, NULL /* allocator */);
	}
}

enum modelpipelinestage {
	DESCRIPTOR_SET_LAYOUT,
	LAYOUT,
	VERT_SHADER,
	FRAG_SHADER,
	SHADERS,
	PIPELINE,
	VERTEX_BUFFER,
	TRANSFORM_BUFFER,
	MODEL_BUFFER,
	DEVICE_MEMORY,
	UNMAP_TRANSFORMS,
	MODEL_PIPELINE_ALL,
};

static void
cleanup_modelpipeline(enum modelpipelinestage stage)
{
	switch (stage) {
	case MODEL_PIPELINE_ALL:
	case UNMAP_TRANSFORMS:
		vkUnmapMemory(device, modelpipeline.memory);
	case DEVICE_MEMORY:
		vkFreeMemory(device, modelpipeline.memory,
			NULL /* allocator */);
	case MODEL_BUFFER:
		vkDestroyBuffer(device, modelpipeline.modelbuf,
			NULL /* allocator */);
	case TRANSFORM_BUFFER:
		vkDestroyBuffer(device, modelpipeline.transbuf,
			NULL /* allocator */);
	case VERTEX_BUFFER:
		vkDestroyBuffer(device, modelpipeline.vertbuf,
			NULL /* allocator */);
	case PIPELINE:
		vkDestroyPipeline(device, modelpipeline.pipeline,
			NULL /* allocator */);
	case SHADERS:
	case FRAG_SHADER:
		vkDestroyShaderModule(device, modelpipeline.fragshader,
			NULL /* allocator */);
	case VERT_SHADER:
		vkDestroyShaderModule(device, modelpipeline.vertshader,
			NULL /* allocator */);
	case LAYOUT:
		vkDestroyPipelineLayout(device, modelpipeline.layout,
			NULL /* allocator */);
	case DESCRIPTOR_SET_LAYOUT:
		vkDestroyDescriptorSetLayout(device, modelpipeline.dsetlayout,
			NULL /* allocator */);
	}
}

enum rendererstage {
	DESCRIPTOR_POOL,
	DEPTH_BUFFER,
	DEPTH_MEMORY,
	DEPTH_VIEW,
	FRAMEBUFFERS,
	RENDER_PASS,
	POOL,
	ACQUIRE_SEMAPHORE,
	COMMAND_BUFFER,
	FENCE,
	RENDERER_ALL,
};

static void
cleanup_renderer(enum rendererstage s)
{
	switch (s) {
	case RENDERER_ALL:
	case FENCE:
		vkDestroyFence(device, renderer.fence, NULL /* allocator */);
	case COMMAND_BUFFER:
		vkFreeCommandBuffers(device, renderer.cpool, 1,
			&renderer.cmdbuf);
	case ACQUIRE_SEMAPHORE:
		vkDestroySemaphore(device, renderer.acquiresem,
			NULL /* allocator */);
	case POOL:
		vkDestroyCommandPool(device, renderer.cpool,
			NULL /* allocator */);
	case FRAMEBUFFERS:
		for (size_t i = 0; i < display.nfbs; i++)
			vkDestroyFramebuffer(device, display.framebuffers[i],
				NULL /* allocator */);
		display.nfbs = 0;
	case RENDER_PASS:
		vkDestroyRenderPass(device, renderer.pass,
			NULL /* allocator */);
	case DEPTH_VIEW:
		vkDestroyImageView(device, renderer.mdview,
			NULL /* allocator */);
	case DEPTH_MEMORY:
		vkFreeMemory(device, renderer.memory, NULL /* allocator */);
	case DEPTH_BUFFER:
		vkDestroyImage(device, renderer.modeldepth,
			NULL /* allocator */);
	case DESCRIPTOR_POOL:
		vkDestroyDescriptorPool(device, renderer.dpool,
			NULL /* allocator */);
	}
}

enum gpustage {
	DISPLAY,
	RENDERER,
	MODEL_PIPELINE,
	END,
};

static void
cleanup_before(enum gpustage stage)
{
	switch (stage) {
		VkQueue queue;
	case END:
		vkGetDeviceQueue(device, qfamidx, 0, &queue);
		vkQueueWaitIdle(queue);
		vkDeviceWaitIdle(device);
		cleanup_modelpipeline(MODEL_PIPELINE_ALL);
	case MODEL_PIPELINE:
		cleanup_renderer(RENDERER_ALL);
	case RENDERER:
		cleanup_display(DISPLAY_ALL);
	case DISPLAY:
		break;
	}
}

static VkShaderModule
read_shader(const char *filename, enum modelpipelinestage toclean)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		err(1, "couldn't open shader file for reading");
	}
	if (fseek(f, 0, SEEK_END) < 0) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		err(1, "couldn't seek to end of shader file");
	}
	size_t codesz = ftell(f);
	if (fseek(f, 0, SEEK_SET) < 0) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		err(1, "couldn't seek to start of shader file");
	}
	uint32_t *code = malloc(codesz);
	if (code == NULL) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		err(1, "couldn't allocate for shader code");
	}
	/* TODO: is endianness an issue here? */
	size_t num = fread(code, 1, codesz, f);
	if (num != codesz) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		if (ferror(f))
			err(1, "couldn't read shader code into buffer");
	}
	fclose(f);

	VkShaderModule shader;
	VkResult res;
	res = vkCreateShaderModule(device, &(VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.codeSize = codesz,
		.pCode = code,
	}, NULL /* allocator */, &shader);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(toclean);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't create shader module: %s", vkstrerror(res));
	}

	return shader;
}

static void
keycb(GLFWwindow *win, int key, int scancode, int action, int mods)
{
	(void)scancode;
	(void)mods;
	if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)
			&& action == GLFW_PRESS)
		glfwSetWindowShouldClose(win, GLFW_TRUE);
	if (key == GLFW_KEY_R)
		reset_camera();
}

static void
mousecb(GLFWwindow *win, int button, int action, int mods)
{
	switch (button) {
	case GLFW_MOUSE_BUTTON_LEFT:
		if (action == GLFW_RELEASE) {
			panning = false;
			rotating = false;
		}

		if ((action == GLFW_PRESS) && (mods & GLFW_MOD_SHIFT)) {
			double x, y;
			glfwGetCursorPos(win, &x, &y);
			lastmouse[0] = x;
			lastmouse[1] = y;
			panning = true;
		}
		if ((action == GLFW_PRESS) && (mods & GLFW_MOD_CONTROL)) {
			double x, y;
			glfwGetCursorPos(win, &x, &y);
			lastmouse[0] = x;
			lastmouse[1] = y;
			rotating = true;
		}
		break;
	}
}

static void
cart2sph(vec3 sph, vec3 cart)
{
	double x = cart[0];
	double y = cart[1];
	double z = cart[2];

	double r = sqrt(x*x + y*y + z*z);
	double t = atan2(x, z);
	double p = asin(y / r);

	sph[0] = r;
	sph[1] = t;
	sph[2] = p;
}

static void
sph2cart(vec3 cart, vec3 sph)
{
	double r = sph[0];
	double t = sph[1];
	double p = sph[2];

	cart[0] = r * cos(p) * sin(t);
	cart[1] = r * sin(p);
	cart[2] = r * cos(p) * cos (t);
}

static void
scrollcb(GLFWwindow *win, double x, double y)
{
	(void)win;
	(void)x;
	vec3 shifted, shiftedP;
	vec3_sub(shifted, camera, target);
	cart2sph(shiftedP, shifted);
	shiftedP[0] += y < 0 ? zoomspeed : -zoomspeed;
	sph2cart(shifted, shiftedP);
	vec3_add(camera, target, shifted);
}

static void
cursorcb(GLFWwindow *win, double x, double y)
{
	(void)win;
	if (panning) {
		vec2 dpos;
		vec2_sub(dpos, lastmouse, (vec2){ x, y });
		lastmouse[0] = x;
		lastmouse[1] = y;

		vec3 f, u, r;
		vec3_sub(f, target, camera);
		vec3_mul_cross(u, f, up);
		vec3_mul_cross(r, f, u);
		vec3_norm(u, u);
		vec3_norm(r, r);
		vec3_scale(u, u, panspeed * dpos[0]);
		vec3_scale(r, r, -panspeed * dpos[1]);

		vec3 diff = { 0, };
		vec3_add(diff, diff, u);
		vec3_add(diff, diff, r);
		vec3_add(camera, camera, diff);
		vec3_add(target, target, diff);
	}

	if (rotating) {
		vec2 dpos;
		vec2_sub(dpos, lastmouse, (vec2){ x, y });
		lastmouse[0] = x;
		lastmouse[1] = y;

		/* TODO: Fix the discontinuities */
		vec3 shifted, shiftedP;
		vec3_sub(shifted, camera, target);
		cart2sph(shiftedP, shifted);
		shiftedP[1] += up[1] * rotspeed * dpos[0];
		shiftedP[2] += up[1] * rotspeed * dpos[1];
		if (shiftedP[2] < -PI/2) {
			shiftedP[1] += PI;
			shiftedP[2] = -PI/2 + (-PI/2 - shiftedP[2]);
			up[1] *= -1;
		} else if (shiftedP[2] > PI/2) {
			shiftedP[1] += PI;
			shiftedP[2] = PI/2 - (shiftedP[2] - PI/2);
			up[1] *= -1;
		}
		if (shiftedP[1] < 0)
			shiftedP[1] += 2*PI;
		else if (shiftedP[1] > 2*PI)
			shiftedP[1] -= 2*PI;
		sph2cart(shifted, shiftedP);
		vec3_add(camera, target, shifted);
	}
}

static void
expand_box(struct boundingbox *box, vec3 point, size_t axis)
{
	if (box->extent[axis] < 0) {
		box->offset[axis] = point[axis];
		box->extent[axis] = 0;
	} else if (point[axis] < box->offset[axis]) {
		box->extent[axis] += box->offset[axis] - point[axis];
		box->offset[axis] = point[axis];
	} else if (point[axis] > box->offset[axis] + box->extent[axis])
		box->extent[axis] = point[axis] - box->offset[axis];
}

/* TODO: decouple objects from meshes */
static void
setup_cpu(void)
{
	const char *files[] = {
		"teapot.norm",
	};

	nobjs = ARR_SZ(files);
	objs = malloc(sizeof(struct object) * nobjs);
	if (objs == NULL) err(1, "couldn't allocate objects");
	bounds = malloc(sizeof(struct bounds) * nobjs);
	if (bounds == NULL) err(1, "couldn't allocate bounding boxes");

	FILE *fs[ARR_SZ(files)];
	for (size_t i = 0; i < ARR_SZ(fs); i++) {
		fs[i] = fopen(files[i], "r");
		if (fs[i] == NULL)
			err(1, "couldn't open %s for reading", files[i]);
		if (fscanf(fs[i], "%zu\n", &objs[i].nprims) == EOF)
			err(1, "couldn't read nprims form %s", files[i]);
		objs[i].meshidx = meshcnt;
		meshcnt += 3*objs[i].nprims;
		/* TODO: alignment is determined by the GPU */
	}

	meshes = malloc(sizeof(struct vertexinfo) * meshcnt);
	if (meshes == NULL) err(1, "couldn't allocate meshes");
	for (size_t i = 0; i < ARR_SZ(fs); i++) {
		struct vertexinfo *vi = meshes + objs[i].meshidx;
		struct boundingbox b = {
			.offset = { 0, 0, 0 },
			.extent = { -1, -1, -1 },
			.dir = { 1.0, 0.0, 0.0 },
		};
		for (size_t j = 0; j < objs[i].nprims; j++) {
			for (size_t k = 0; k < 3; k++) {
				float *vert = (float *)(vi + 3*j + k);
				int res = fscanf(fs[i], "%f %f %f\n%f %f %f\n",
					vert, vert+1, vert+2,
					vert+3, vert+4, vert+5);
				if (res == EOF) {
					if (ferror(fs[i]))
						err(1, "couldn't read data from"
							" %s", files[i]);
					errx(1, "couldn't read data from %s:"
						" unexpected EOF", files[i]);
				}
				expand_box(&b, vert, 0);
				expand_box(&b, vert, 1);
				expand_box(&b, vert, 2);
			}
			if (fscanf(fs[i], "\n") == EOF) {
				if (ferror(fs[i]))
					err(1, "couldn't read data from %s",
						files[i]);
				errx(1, "couldn't read data from %s:"
					" unexpected EOF", files[i]);
			}
		}
		fclose(fs[i]);

		bounds[i].orig = b;
		vec4_scale(bounds[i].curr.offset, b.extent, -20);
		vec4_scale(bounds[i].curr.extent, b.extent, 40);
	}

	reset_camera();
}

static void
setup_display(void)
{
	VkResult res;

	if (!glfwInit()) {
		const char *msg;
		(void)glfwGetError(&msg);
		errx(1, "couldn't initialise GLFW context: %s", msg);
	}

	/* TODO: set the error callback and figure out how I want to use it */

	if (!glfwVulkanSupported())
		errx(1, "vulkan not supported by GLFW");

	uint32_t apiver;
	res = vkEnumerateInstanceVersion(&apiver);
	if (res != VK_SUCCESS)
		errx(1, "coudln't determine vulkan API version");
	if (VK_API_VERSION_MAJOR(apiver) != 1 ||
			VK_API_VERSION_MINOR(apiver) < 1)
		errx(1, "requires compatibility with version 1.1 of vulkan");

	const char *manualexts[] = {
		"VK_KHR_surface",
		"VK_KHR_get_surface_capabilities2",
		"VK_EXT_layer_settings",
	};
	uint32_t numglfwexts;
	const char **glfwexts = glfwGetRequiredInstanceExtensions(&numglfwexts);
	if (numglfwexts == 0) {
		const char *msg;
		(void)glfwGetError(&msg);
		errx(1, "couldn't get required vulkan extensions: %s", msg);
	}
	if (SIZE_MAX - numglfwexts < ARR_SZ(manualexts))
		errx(1, "can't allocate more than SIZE_MAX (%zu) extensions:"
			" GLFW requested %zu and we requested %zu", numglfwexts,
			ARR_SZ(manualexts));
	size_t nexts = numglfwexts + ARR_SZ(manualexts);
	if (SIZE_MAX / sizeof(char **) < nexts)
		errx(1, "can't allocate space for %zu extensions"
			" (%zu * sizeof(char **) > SIZE_MAX)", nexts);

	const char **exts = malloc(sizeof(char **) * nexts);
	if (exts == NULL)
		err(1, "couldn't allocate for instance extensions");
	memcpy(exts, glfwexts, sizeof(char *) * numglfwexts);
	memcpy(exts + numglfwexts, manualexts, sizeof(manualexts));
	res = vkCreateInstance(&(VkInstanceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = &(VkValidationFeaturesEXT) {
			.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
			.pNext = NULL,
			.enabledValidationFeatureCount = 1,
			.pEnabledValidationFeatures = (VkValidationFeatureEnableEXT[]){
				VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
			},
			.disabledValidationFeatureCount = 0,
			.pDisabledValidationFeatures = NULL,
		},
		.flags = 0,
		.pApplicationInfo = &(VkApplicationInfo) {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pNext = NULL,
			.pApplicationName = progname,
			.applicationVersion = 0,
			.pEngineName = 0,
			.engineVersion = 0,
			.apiVersion = VK_MAKE_API_VERSION(0, 1, 1, 0),
		},
		.enabledLayerCount = 1,
		.ppEnabledLayerNames = (const char *[]){
			"VK_LAYER_KHRONOS_validation",
		},
		.enabledExtensionCount = numglfwexts + ARR_SZ(manualexts),
		.ppEnabledExtensionNames = exts,
	}, NULL /* allocator */, &instance);
	free(exts);
	if (res != VK_SUCCESS)
		errx(1, "couldn't create vulkan instance: %s", vkstrerror(res));

	/* TODO: allow resizing and window decorations */
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
	display.window = glfwCreateWindow(
		width, height, progname, glfwGetPrimaryMonitor(), NULL);
	if (display.window == NULL) {
		const char *msg;
		(void)glfwGetError(&msg);
		glfwTerminate();
		cleanup_display(INSTANCE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't create a GLFW window: %s", msg);
	}

	res = glfwCreateWindowSurface(instance, display.window,
		NULL /* allocator */, &display.surface);
	if (res != VK_SUCCESS) {
		cleanup_display(INSTANCE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't create surface for window: %s",
			vkstrerror(res));
	}
	glfwSetKeyCallback(display.window, &keycb);
	glfwSetMouseButtonCallback(display.window, &mousecb);
	glfwSetScrollCallback(display.window, &scrollcb);
	glfwSetCursorPosCallback(display.window, &cursorcb);

	uint32_t devcnt;
	res = vkEnumeratePhysicalDevices(instance, &devcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		errx(1, "coudln't get number of physical devices: %s",
			vkstrerror(res));
	}
	if (SIZE_MAX / sizeof(VkPhysicalDevice) < (size_t)devcnt ||
			(size_t)devcnt != devcnt) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate space for %zu devices"
			" (%zu * sizeof(VkPhysicalDevice) > SIZE_MAX)", devcnt);
	}
	VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * devcnt);
	if (devs == NULL) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate to store physical devices");
	}
	size_t devcap = devcnt;
	while ((res = vkEnumeratePhysicalDevices(instance, &devcnt, devs))
			== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkPhysicalDevice) > devcap) {
			cleanup_display(SURFACE);
			cleanup_before(DISPLAY);
			errx(1, "can't store %zu physical devices", devcap);
		}
		devcap *= 2;
		VkPhysicalDevice *tmp = realloc(
			devs, sizeof(VkPhysicalDevice) * devcap);
		if (tmp == NULL) {
			cleanup_display(SURFACE);
			cleanup_before(DISPLAY);
			err(1, "coudln't allocate to store physical devices");
		}
		devs = tmp;
		devcnt = devcap;
	}
	if (res != VK_SUCCESS) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't get physical devices: %s", vkstrerror(res));
	}

	VkPhysicalDevice physdev;
	VkPhysicalDeviceProperties2 devprops;
	bool found = false;
	for (size_t i = 0; i < devcnt; i++) {
		VkPhysicalDeviceProperties2 props = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = NULL,
		};
		vkGetPhysicalDeviceProperties2(devs[i], &props);
		uint32_t qidx, midx;
		if (!usabledev(devs[i], props, &qidx, &midx))
			continue;

		if (!found || preferdev(props, devprops)) {
			physdev = devs[i];
			devprops = props;
			qfamidx = qidx;
			memidx = midx;
			found = true;
			continue;
		}
	}
	free(devs);

	if (!found) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't find a compatible GPU");
	}

	const char *devexts[] = {
		"VK_KHR_swapchain",
	};
	res = vkCreateDevice(physdev, &(VkDeviceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = (const VkDeviceQueueCreateInfo[]){ {
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.pNext = NULL,
				.flags = 0,
				.queueFamilyIndex = qfamidx,
				.queueCount = 1,
				.pQueuePriorities = (const float[]) { 0.0 },
			},
		},
		.enabledExtensionCount = ARR_SZ(devexts),
		.ppEnabledExtensionNames = devexts,
		.pEnabledFeatures = NULL,
	}, NULL /* allocator */, &device);
	if (res != VK_SUCCESS) {
		cleanup_display(SURFACE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't create logical device: %s", vkstrerror(res));
	}

	VkSurfaceCapabilitiesKHR caps;
	res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
		physdev, display.surface, &caps);
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't query for surface capabilities: %s",
			vkstrerror(res));
	}

	VkPhysicalDeviceSurfaceInfo2KHR surfinfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
		.pNext = NULL,
		.surface = display.surface,
	};
	uint32_t fmtcnt;
	res = vkGetPhysicalDeviceSurfaceFormats2KHR(
		physdev, &surfinfo, &fmtcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't query for surface formats: %s",
			vkstrerror(res));
	}
	size_t fmtcap = fmtcnt;
	if (SIZE_MAX / sizeof(VkSurfaceFormat2KHR) < fmtcap) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate %zu surface formats");
	}
	VkSurfaceFormat2KHR *fmts = malloc(sizeof(VkSurfaceFormat2KHR)*fmtcap);
	if (fmts == NULL) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate for surface formats");
	}
	for (size_t i = 0; i < fmtcnt; i++)
		fmts[i] = (VkSurfaceFormat2KHR){
			.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
			.pNext = NULL,
		};
	while ((res = vkGetPhysicalDeviceSurfaceFormats2KHR(
			physdev, &surfinfo, &fmtcnt, fmts))
			== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkSurfaceFormat2KHR) > fmtcap) {
			cleanup_display(DEVICE);
			cleanup_before(DISPLAY);
			errx(1, "can't store %zu physical devices", fmtcap);
		}
		fmtcap *= 2;
		VkSurfaceFormat2KHR *tmp = realloc(
			fmts, sizeof(VkSurfaceFormat2KHR) * fmtcap);
		if (tmp == NULL) {
			cleanup_display(DEVICE);
			cleanup_before(DISPLAY);
			err(1, "coudln't allocate to store surface formats");
		}
		fmts = tmp;
		for (size_t i = fmtcnt; i < fmtcap; i++)
			fmts[i] = (VkSurfaceFormat2KHR){
				.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
				.pNext = NULL,
			};
		fmtcnt = fmtcap;
	}
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't get surface formats: %s", vkstrerror(res));
	}

	uint32_t modecnt;
	res = vkGetPhysicalDeviceSurfacePresentModesKHR(
		physdev, display.surface, &modecnt, NULL);
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't query for surface formats: %s",
			vkstrerror(res));
	}
	size_t modecap = modecnt;
	if (SIZE_MAX / sizeof(VkPresentModeKHR) < modecap) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate %zu present modes", modecap);
	}
	VkPresentModeKHR *modes = malloc(sizeof(VkPresentModeKHR)*modecap);
	if (modes == NULL) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate for surface formats");
	}
	while ((res = vkGetPhysicalDeviceSurfacePresentModesKHR(
			physdev, display.surface, &modecnt, modes))
			== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkPresentModeKHR) > modecap) {
			cleanup_display(DEVICE);
			cleanup_before(DISPLAY);
			errx(1, "can't store %zu physical devices", modecap);
		}
		modecap *= 2;
		VkPresentModeKHR *tmp = realloc(
			modes, sizeof(VkPresentModeKHR) * modecap);
		if (tmp == NULL) {
			cleanup_display(DEVICE);
			cleanup_before(DISPLAY);
			err(1, "coudln't allocate to store surface formats");
		}
		modes = tmp;
		modecnt = modecap;
	}
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't get surface formats: %s", vkstrerror(res));
	}

	display.caps = caps;
	display.fmt = fmts[0].surfaceFormat;
	display.mode = modes[0];
	free(fmts);
	free(modes);

	res = vkCreateSwapchainKHR(device, &(VkSwapchainCreateInfoKHR){
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = NULL,
		.flags = 0,
		.surface = display.surface,
		.minImageCount = display.caps.minImageCount,
		.imageFormat = display.fmt.format,
		.imageColorSpace = display.fmt.colorSpace,
		/* TODO: check caps */
		.imageExtent = {
			.width = width,
			.height = height,
		},
		.imageArrayLayers = 1,
		/* TODO: check caps */
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (const uint32_t[]){ qfamidx },
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		/* TODO: check caps */
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = display.mode,
		.clipped = VK_TRUE,
		.oldSwapchain = VK_NULL_HANDLE,
	}, NULL /* allocator */, &display.swapchain);
	if (res != VK_SUCCESS) {
		cleanup_display(DEVICE);
		cleanup_before(DISPLAY);
		errx(1, "couldn't create swapchain: %s", vkstrerror(res));
	}

	uint32_t imcnt;
	res = vkGetSwapchainImagesKHR(device, display.swapchain, &imcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanup_display(SWAP_CHAIN);
		cleanup_before(DISPLAY);
		errx(1, "couldn't get swapchain images: %s", vkstrerror(res));
	}
	size_t imcap = imcnt;
	if (SIZE_MAX / sizeof(VkImage) < imcap) {
		cleanup_display(SWAP_CHAIN);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate %zu images", imcap);
	}
	VkImage *ims = malloc(sizeof(VkImage) * imcap);
	if (ims == NULL) {
		cleanup_display(SWAP_CHAIN);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate memory for images");
	}
	while ((res = vkGetSwapchainImagesKHR(
			device, display.swapchain, &imcnt, ims))
			== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkImage) > imcap) {
			cleanup_display(SWAP_CHAIN);
			cleanup_before(DISPLAY);
			errx(1, "can't store %zu physical devices", imcap);
		}
		imcap *= 2;
		VkImage *tmp = realloc(modes, sizeof(VkImage) * imcap);
		if (tmp == NULL) {
			cleanup_display(SWAP_CHAIN);
			cleanup_before(DISPLAY);
			err(1, "coudln't allocate to store surface formats");
		}
		ims = tmp;
		imcnt = imcap;
	}
	display.images = ims;
	display.nims = imcnt;

	if (SIZE_MAX / sizeof(VkSemaphore) < (size_t)imcnt ||
			(size_t)imcnt != imcnt) {
		cleanup_display(SWAP_CHAIN);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate %zu semaphores", imcnt);
	}
	display.semaphores = malloc(sizeof(VkSemaphore) * imcnt);
	if (display.semaphores == NULL) {
		cleanup_display(SWAP_CHAIN);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate for semaphores");
	}
	display.nsems = 0;
	for (size_t i = 0; i < display.nims; i++) {
		res = vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
		}, NULL /* allocator */, display.semaphores+i);
		if (res != VK_SUCCESS) {
			cleanup_display(SEMAPHORES);
			cleanup_before(DISPLAY);
			errx(1, "couldn't create semaphore: %s", vkstrerror(res));
		}
		display.nsems++;
	}

	if (SIZE_MAX / sizeof(VkImageView) < (size_t)display.nims ||
			(size_t)display.nims != display.nims) {
		cleanup_display(SEMAPHORES);
		cleanup_before(DISPLAY);
		errx(1, "can't allocate %zu image views", display.nims);
	}
	display.views = malloc(sizeof(VkImageView) * display.nims);
	if (display.views == NULL) {
		cleanup_display(SEMAPHORES);
		cleanup_before(DISPLAY);
		err(1, "couldn't allocate for image views");
	}
	display.nviews = 0;
	for (size_t i = 0; i < imcnt; i++) {
		res = vkCreateImageView(device, &(VkImageViewCreateInfo){
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.image = display.images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = display.fmt.format,
			.components = {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		}, NULL /* allocator */, display.views+i);
		if (res != VK_SUCCESS) {
			cleanup_display(IMAGE_VIEWS);
			cleanup_before(DISPLAY);
			errx(1, "couldn't create image view: %s", vkstrerror(res));
		}
		display.nviews++;
	}
}

static void
setup_modelpipeline(void)
{
	VkResult res;

	res = vkCreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = 2,
		.pBindings = (VkDescriptorSetLayoutBinding[]){ {
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.pImmutableSamplers = NULL,
			}, {
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.pImmutableSamplers = NULL,
			},
		},
	}, NULL /* allocator */, &modelpipeline.dsetlayout);
	if (res != VK_SUCCESS) {
		cleanup_renderer(RENDERER_ALL);
		cleanup_before(RENDERER);
		errx(1, "couldn't allocate descriptor set layout: %s",
			vkstrerror(res));
	}

	res = vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			modelpipeline.dsetlayout,
		},
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL,
	}, NULL /* allocator */, &modelpipeline.layout);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(DESCRIPTOR_SET_LAYOUT);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't create pipeline layout: %s", vkstrerror(res));
	}

	modelpipeline.vertshader = read_shader("vert.spv", LAYOUT);
	modelpipeline.fragshader = read_shader("frag.spv", VERT_SHADER);

	res = vkCreateGraphicsPipelines(device, NULL /* cache */, 1,
			&(VkGraphicsPipelineCreateInfo){
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		/* TODO: */ .flags = 0,
		.stageCount = 2,
		.pStages = (VkPipelineShaderStageCreateInfo[]){ {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.pNext = NULL,
				.flags = 0,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = modelpipeline.vertshader,
				.pName = "main",
				.pSpecializationInfo = NULL,
			}, {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 	.pNext = NULL,
			 	.flags = 0,
			 	.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			 	.module = modelpipeline.fragshader,
			 	.pName = "main",
			 	.pSpecializationInfo = NULL,
			},
		},
		.pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
				.binding = 0,
				.stride = sizeof(struct vertexinfo),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
			.vertexAttributeDescriptionCount = 2,
			.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){{
					.binding = 0,
					.location = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(struct vertexinfo, pos),
				}, {
					.binding = 0,
					.location = 1,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(struct vertexinfo, norm),
				},
			},
		},
		.pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		},
		.pTessellationState = &(VkPipelineTessellationStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			/* TODO: */ .patchControlPoints = 1,
		},
		.pViewportState = &(VkPipelineViewportStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.pNext = 0,
			.flags = 0,
			.viewportCount = 1,
			.pViewports = &(VkViewport){
				.x = (1-meshpct)*width,
				.y = 0,
				.width = meshpct * width,
				.height = height,
				.minDepth = 0.0,
				.maxDepth = 1.0,
			},
			.scissorCount = 1,
			.pScissors = &(VkRect2D){
				.offset = (VkOffset2D){
					.x = (1-meshpct)*width,
					.y = 0
				},
				.extent = (VkExtent2D){
					.width = meshpct * width,
					.height = height
				},
			},
		},
		.pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.depthBiasConstantFactor = 0,
			.depthBiasClamp = 0,
			.depthBiasSlopeFactor = 0,
			/* TODO: */ .lineWidth = 1.0,
		},
		.pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
			.minSampleShading = 0,
			.pSampleMask = NULL,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable = VK_FALSE,
		},
		.pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_GREATER,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
			.front = { 0 },
			.back = { 0 },
			.minDepthBounds = 0.0,
			.maxDepthBounds = 1.0,
		},
		.pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.logicOpEnable = VK_FALSE,
			.logicOp = 0,
			.attachmentCount = 1,
			.pAttachments = &(VkPipelineColorBlendAttachmentState){
				.blendEnable = VK_FALSE,
				.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR,
				.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR,
				.colorBlendOp = VK_BLEND_OP_ADD,
				.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_COLOR,
				.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_COLOR,
				.alphaBlendOp = VK_BLEND_OP_ADD,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
					VK_COLOR_COMPONENT_G_BIT |
					VK_COLOR_COMPONENT_B_BIT |
					VK_COLOR_COMPONENT_A_BIT,
			},
			.blendConstants = { 0, 0, 0, 0 },
		},
		.pDynamicState = NULL,
		.layout = modelpipeline.layout,
		.renderPass = renderer.pass,
		.subpass = 0,
		.basePipelineHandle = 0,
		.basePipelineIndex = 0,
	}, NULL /* allocator */, &modelpipeline.pipeline);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(SHADERS);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't create graphics pipeline: %s",
			vkstrerror(res));
	}

	/* TODO: put these bad bois in variables (size) */
	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(struct vertexinfo) * meshcnt,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
		},
	}, NULL /* allocator */, &modelpipeline.vertbuf);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(PIPELINE);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(mat4x4) * nobjs,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
		},
	}, NULL /* allocator */, &modelpipeline.transbuf);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(VERTEX_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(mat4x4) * nobjs,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
			},
	}, NULL /* allocator */, &modelpipeline.modelbuf);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(TRANSFORM_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	/* TODO: all of this can overflow */
	VkMemoryRequirements memreq;
	vkGetBufferMemoryRequirements(device, modelpipeline.vertbuf, &memreq);
	size_t align = memreq.alignment - 1;
	meshsz = (sizeof(struct vertexinfo)*meshcnt + align) & ~align;
	size_t transsz = (sizeof(mat4x4) * nobjs + align) & ~align;
	size_t modelsz = (sizeof(mat4x4) * nobjs + align) & ~align;
	size_t allocsz = max(memreq.size, meshsz + transsz + modelsz);
	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = allocsz,
		.memoryTypeIndex = memidx,
	}, NULL /* allocator */, &modelpipeline.memory);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(MODEL_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate device memory: %s", vkstrerror(res));
	}

	res = vkBindBufferMemory2(device, 3, (VkBindBufferMemoryInfo[]){{
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = modelpipeline.vertbuf,
			.memory = modelpipeline.memory,
			.memoryOffset = 0,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = modelpipeline.transbuf,
			.memory = modelpipeline.memory,
			.memoryOffset = meshsz,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = modelpipeline.modelbuf,
			.memory = modelpipeline.memory,
			.memoryOffset = meshsz + transsz,
		},
	});
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't bind buffer memory: %s", vkstrerror(res));
	}

	/* TODO: investigate the consequences of this being one big mem block */
	void *ptr;
	res = vkMapMemory(device, modelpipeline.memory, 0, meshsz, 0, &ptr); 
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	memcpy(ptr, meshes, sizeof(struct vertexinfo)*meshcnt);
	vkUnmapMemory(device, modelpipeline.memory);

	res = vkMapMemory(device, modelpipeline.memory, meshsz, transsz+modelsz,
		0, (void **)&transforms);
	if (res != VK_SUCCESS) {
		cleanup_modelpipeline(DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	models = (mat4x4 *)((char *)transforms + transsz);
}

static void
setup_renderer(void)
{
	VkResult res;

	res = vkCreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.maxSets = 1,
		.poolSizeCount = 2,
		.pPoolSizes = (VkDescriptorPoolSize[]){ {
				.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
			}, {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 2,
			},
		},
	}, NULL /* allocator */, &renderer.dpool);
	if (res != VK_SUCCESS) {
		cleanup_before(RENDERER);
		errx(1, "coudln't create descriptor pool: %s", vkstrerror(res));
	}


	res = vkCreateImage(device, &(VkImageCreateInfo){
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_D32_SFLOAT,
		.extent = {
			.width = width,
			.height = height,
			.depth = 1,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]){ qfamidx },
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	}, NULL /* allocator */, &renderer.modeldepth);
	if (res != VK_SUCCESS) {
		cleanup_renderer(DESCRIPTOR_POOL);
		cleanup_before(RENDERER);
		errx(1, "couldn't create image for depth buffer: %s",
			vkstrerror(res));
	}

	VkMemoryRequirements2 imreqs = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
		.pNext = NULL,
	};
	vkGetImageMemoryRequirements2(device, &(VkImageMemoryRequirementsInfo2){
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
		.pNext = NULL,
		.image = renderer.modeldepth,
	}, &imreqs);
	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		/* TODO: align */
		.allocationSize = imreqs.memoryRequirements.size,
		.memoryTypeIndex = memidx,
	}, NULL /* allocator */, &renderer.memory);
	if (res != VK_SUCCESS) {
		cleanup_renderer(DEPTH_BUFFER);
		cleanup_before(RENDERER);
		errx(1, "couldn't allocate memory for depth buffer: %s",
			vkstrerror(res));
	}

	res = vkBindImageMemory2(device, 1, (VkBindImageMemoryInfo[]){{
			.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
			.pNext = NULL,
			.image = renderer.modeldepth,
			.memory = renderer.memory,
			.memoryOffset = 0,
		},
	});
	if (res != VK_SUCCESS) {
		cleanup_renderer(DEPTH_MEMORY);
		cleanup_before(RENDERER);
		errx(1, "couldn't bind buffer memory: %s", vkstrerror(res));
	}

	res = vkCreateImageView(device, &(VkImageViewCreateInfo){
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.image = renderer.modeldepth,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_D32_SFLOAT,
		.components = { 0, },
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		}
	}, NULL /* allocator */, &renderer.mdview);
	if (res != VK_SUCCESS) {
		cleanup_renderer(DEPTH_MEMORY);
		cleanup_before(RENDERER);
		errx(1, "couldn't create image view for depth buffer: %s",
			vkstrerror(res));
	}

	res = vkCreateRenderPass(device, &(VkRenderPassCreateInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = 2,
		.pAttachments = (const VkAttachmentDescription[]){ {
				.flags = 0,
				.format = display.fmt.format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			}, {
				.flags = 0,
				.format = VK_FORMAT_D32_SFLOAT,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
		},
		.subpassCount = 1,
		.pSubpasses = &(VkSubpassDescription){
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = 1,
			.pColorAttachments = (VkAttachmentReference[]){ {
					.attachment = 0,
					.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				},
			 },
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = (VkAttachmentReference[]){ {
					.attachment = 1,
					.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				},
			},
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL,
		},
		.dependencyCount = 1,
		.pDependencies = (VkSubpassDependency[]){{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			},
		},
	}, NULL /* allocator */, &renderer.pass);
	if (res != VK_SUCCESS) {
		cleanup_renderer(DEPTH_VIEW);
		cleanup_before(RENDERER);
		errx(1, "coudln't create render pass: %s", vkstrerror(res));
	}

	display.framebuffers = malloc(sizeof(VkFramebuffer) * display.nims);
	if (display.framebuffers == NULL) {
		cleanup_renderer(RENDER_PASS);
		cleanup_before(RENDERER);
		err(1, "couldn't not allocate for framebuffers");
	}
	display.nfbs = 0;
	for (size_t i = 0; i < display.nims; i++) {
		res = vkCreateFramebuffer(device, &(VkFramebufferCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.renderPass = renderer.pass,
			.attachmentCount = 2,
			.pAttachments = (VkImageView[]) {
				display.views[i],
				renderer.mdview,
			},
			.width = width,
			.height = height,
			.layers = 1,
		}, NULL /* allocator */, display.framebuffers+i);
		if (res != VK_SUCCESS) {
			cleanup_renderer(FRAMEBUFFERS);
			cleanup_before(RENDERER);
			errx(1, "coudln't allocate frame buffer: %s",
				vkstrerror(res));
		}
		display.nfbs++;
	}

	res = vkCreateCommandPool(device, &(VkCommandPoolCreateInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = qfamidx,
	}, NULL /* allocator */, &renderer.cpool);
	if (res != VK_SUCCESS) {
		cleanup_renderer(FRAMEBUFFERS);
		cleanup_before(RENDERER);
		errx(1, "couldn't allocate command pool: %s", vkstrerror(res));
	}
	res = vkCreateFence(device, &(VkFenceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
	}, NULL /* allocator */, &renderer.fence);
	if (res != VK_SUCCESS) {
		cleanup_renderer(POOL);
		cleanup_before(RENDERER);
		errx(1, "couldn't create fence: %s", vkstrerror(res));
	}

	res = vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
	}, NULL /* allocator */, &renderer.acquiresem);
	if (res != VK_SUCCESS) {
		cleanup_renderer(FENCE);
		cleanup_before(RENDERER);
		errx(1, "couldn't create semaphore: %s", vkstrerror(res));
	}

	res = vkAllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = renderer.cpool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	}, &renderer.cmdbuf);
	if (res != VK_SUCCESS) {
		cleanup_renderer(ACQUIRE_SEMAPHORE);
		cleanup_before(RENDERER);
		errx(1, "couldn't get a command buffer: %s", vkstrerror(res));
	}
}

static void
render(void)
{
	VkResult res;

	uint32_t idx;
	res = vkAcquireNextImage2KHR(device, &(VkAcquireNextImageInfoKHR){
		.sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
		.pNext = NULL,
		.swapchain = display.swapchain,
		.timeout = 0,
		.semaphore = renderer.acquiresem,
		.fence = VK_NULL_HANDLE,
		.deviceMask = 1,
	}, &idx);
	if (res != VK_SUCCESS) {
		warnx("couldn't acquire image for presentation");
		return;
	}

	VkDescriptorSet dset;
	res = vkAllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = renderer.dpool,
		.descriptorSetCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			modelpipeline.dsetlayout,
		}
	}, &dset);
	if (res != VK_SUCCESS) {
		warnx("couldn't get a descriptor set: %s", vkstrerror(res));
		return;
	}

	res = vkResetCommandBuffer(renderer.cmdbuf, 0);
	if (res != VK_SUCCESS) {
		warnx("couldn't reset command buffer: %s", vkstrerror(res));
		return;
	}

	res = vkBeginCommandBuffer(renderer.cmdbuf, &(VkCommandBufferBeginInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = NULL,
	});
	if (res != VK_SUCCESS) {
		warnx("couldn't begin command buffer: %s", vkstrerror(res));
		goto free_dset;
	}

	for (size_t i = 0; i < nobjs; i++) {
		mat4x4 A, model, view, proj;

		const float *off = bounds[i].orig.offset;
		const float *box = bounds[i].curr.extent;
		const float *ogbox = bounds[i].orig.extent;
		mat4x4_identity(A);
		mat4x4_scale_aniso(model, A,
			box[0] / ogbox[0],
			box[1] / ogbox[1],
			box[2] / ogbox[2]);
		mat4x4_translate(A, off[0], off[1], off[2]);
		mat4x4_mul(model, A, model);
		mat4x4_look_at(view, camera, target, up);
		mat4x4_perspective(proj, 120, width/height, -1, 1);
		mat4x4_mul(A, view, model);
		mat4x4_mul(transforms[i], proj, A);
		mat4x4_dup(models[i], model);
	}
	vkUpdateDescriptorSets(device, 1, (VkWriteDescriptorSet[]){{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = NULL,
			.dstSet = dset,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pImageInfo = NULL,
			.pBufferInfo = (VkDescriptorBufferInfo[]){{
					.buffer = modelpipeline.transbuf,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				}, {
					.buffer = modelpipeline.modelbuf,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				},
			},
			.pTexelBufferView = NULL,
		},
	}, 0, NULL);
	vkCmdBindDescriptorSets(renderer.cmdbuf,
		VK_PIPELINE_BIND_POINT_GRAPHICS, modelpipeline.layout, 0, 1,
		&dset, 2, (uint32_t[]){ 0, 0 });
	vkCmdBindPipeline(renderer.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
		modelpipeline.pipeline);
	vkCmdBindVertexBuffers(renderer.cmdbuf, 0, 1,
		(VkBuffer[]){ modelpipeline.vertbuf },
		(VkDeviceSize[]){ 0 });
	vkCmdBeginRenderPass(renderer.cmdbuf, &(VkRenderPassBeginInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = renderer.pass,
		.framebuffer = display.framebuffers[idx],
		.renderArea = {
			.offset = { .x = (1-meshpct)*width, .y = 0 },
			.extent = { .width = meshpct*width, .height = height },
		},
		.clearValueCount = 2,
		.pClearValues = (VkClearValue[]) { {
				.color = { .float32 = { 0.6, 0.6, 0.6, 1.0 } },
			}, {
				.depthStencil = {
					.depth = 0,
					.stencil = 0,
				},
			}
		},
	}, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdDraw(renderer.cmdbuf, meshcnt, 1, 0, 0);
	vkCmdEndRenderPass(renderer.cmdbuf);
	vkEndCommandBuffer(renderer.cmdbuf);

	VkQueue queue;
	vkGetDeviceQueue(device, qfamidx, 0, &queue);
	res = vkQueueSubmit(queue, 1, &(VkSubmitInfo){
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = (VkSemaphore[]){ renderer.acquiresem },
		.pWaitDstStageMask = (VkPipelineStageFlags[]){
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		},
		.commandBufferCount = 1,
		.pCommandBuffers = (VkCommandBuffer[]){
			renderer.cmdbuf,
		},
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = (VkSemaphore[]){ display.semaphores[idx] },
	}, renderer.fence);
	if (res != VK_SUCCESS)
		warnx("couldn't submit work to queue: %s", vkstrerror(res));

	vkQueuePresentKHR(queue, &(VkPresentInfoKHR){
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = (VkSemaphore[]){ display.semaphores[idx] },
		.swapchainCount = 1,
		.pSwapchains = (VkSwapchainKHR[]){ display.swapchain },
		.pImageIndices = (uint32_t[]){ idx },
		.pResults = NULL,
	});

	res = vkWaitForFences(device, 1, (VkFence[]){ renderer.fence },
		VK_TRUE, (uint64_t)-1);
	if (res != VK_SUCCESS)
		warnx("couldn't wait on fence");
	vkResetFences(device, 1, (VkFence[]){ renderer.fence });

free_dset:
	vkResetDescriptorPool(device, renderer.dpool, 0);
}

int
main(void)
{
	setup_cpu();
	setup_display();
	setup_renderer();
	setup_modelpipeline();
	struct timespec prev = {0};
	while (!glfwWindowShouldClose(display.window)) {
		render();
		glfwWaitEvents();

		struct timespec curr;
		clock_gettime(CLOCK_MONOTONIC, &curr);
		while (nanotimerdiff(curr, prev) < 1e9/60) {
			nanosleep(&(struct timespec){ .tv_nsec = 5e5 }, NULL);
			clock_gettime(CLOCK_MONOTONIC, &curr);
		}
		prev = curr;
	}
	cleanup_before(END);
	glfwDestroyWindow(display.window);
	glfwTerminate();

	return 0;
}
