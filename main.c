#include <err.h>
#include <errno.h>
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
#define STBI_ONLY_TGA
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define ARR_SZ(a) (sizeof(a) / sizeof((a)[0]))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define PI ((double)3.14159265358979323846)

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

struct modelvertex {
	vec3 pos;
	vec3 norm;
};

static struct {
	VkDescriptorSetLayout dsetlayout;
	VkDescriptorSet dset;
	VkPipelineLayout layout;
	VkPipeline pipeline;
	VkShaderModule vertshader;
	VkShaderModule fragshader;
	VkDeviceMemory memory;
	VkBuffer vertbuf;
	VkBuffer transbuf;
	VkBuffer modelbuf;

	size_t nobjs;
	struct object *objs;
	struct modelvertex *meshes;
	size_t meshcnt;
	size_t meshsz;
	struct bounds *bounds;
	vec3 camera;
	vec3 target;
	vec3 up;
	mat4x4 *transforms;
	mat4x4 *models;

	vec2 lastmouse;
	bool panning;
	bool rotating;
} model;

struct uivertex {
	vec2 pos;
};

static struct {
	VkDescriptorSetLayout dsetlayout;
	VkDescriptorSet *dsets;
	VkPipelineLayout layout;
	VkPipeline pipeline;
	VkShaderModule vertshader;
	VkShaderModule fragshader;
	VkDeviceMemory bufmem;
	VkBuffer vertbuf;
	VkBuffer idxbuf;
	VkBuffer iconbuf;
	VkDeviceMemory immem;
	size_t *iconoff;
	VkImage *iconims;
	size_t niconims;
	VkBindImageMemoryInfo *imbinds;
	VkImageView *icons;
	size_t niconviews;
	VkSampler sampler;

	size_t nverts;
	struct uivertex *verts;
	size_t nidxs;
	uint32_t *idxs;
	size_t nicons;
	unsigned char *texels;
	size_t texelsz;
	size_t *texoff;
	vec2 *boxes;
} ui;

static struct {
	VkRenderPass pass;
	VkDescriptorPool dpool;
	VkCommandPool cpool;
	VkFence fence;
	VkSemaphore acquiresem;
	VkCommandBuffer cmdbuf;
	VkDeviceMemory memory;

	VkImage depthim;
	VkImageView depth;
} renderer;

static int64_t
nanotimerdiff(struct timespec a, struct timespec b)
{
	return 1e9*(a.tv_sec-b.tv_sec) + (a.tv_nsec-b.tv_nsec);
}

static void *
nmalloc(size_t size, size_t n)
{
	if (SIZE_MAX / size < n) {
		errno = EINVAL;
		return NULL;
	}

	void *res = malloc(size * n);
	return res;
}

static void
reset_camera(void)
{
	model.target[0] = 0;
	model.target[1] = 0;
	model.target[2] = 0;
	model.up[0] = 0;
	model.up[1] = -1;
	model.up[2] = 0;
	model.camera[0] = 0;
	model.camera[1] = height/4;
	model.camera[2] = depth/2;
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
		nmalloc(sizeof(VkQueueFamilyProperties2), qcnt);
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

enum modelstage {
	MP_DESCRIPTOR_SET_LAYOUT,
	MP_LAYOUT,
	MP_VERT_SHADER,
	MP_FRAG_SHADER,
	MP_SHADERS,
	MP_PIPELINE,
	MP_VERTEX_BUFFER,
	MP_TRANSFORM_BUFFER,
	MP_MODEL_BUFFER,
	MP_DEVICE_MEMORY,
	MP_UNMAP_TRANSFORMS,
	MODEL_PIPELINE_ALL,
};

static void
cleanup_model(enum modelstage stage)
{
	switch (stage) {
	case MODEL_PIPELINE_ALL:
	case MP_UNMAP_TRANSFORMS:
		vkUnmapMemory(device, model.memory);
	case MP_DEVICE_MEMORY:
		vkFreeMemory(device, model.memory, NULL /* allocator */);
	case MP_MODEL_BUFFER:
		vkDestroyBuffer(device, model.modelbuf, NULL /* allocator */);
	case MP_TRANSFORM_BUFFER:
		vkDestroyBuffer(device, model.transbuf, NULL /* allocator */);
	case MP_VERTEX_BUFFER:
		vkDestroyBuffer(device, model.vertbuf, NULL /* allocator */);
	case MP_PIPELINE:
		vkDestroyPipeline(device, model.pipeline, NULL /* allocator */);
	case MP_SHADERS:
	case MP_FRAG_SHADER:
		vkDestroyShaderModule(device, model.fragshader,
			NULL /* allocator */);
	case MP_VERT_SHADER:
		vkDestroyShaderModule(device, model.vertshader,
			NULL /* allocator */);
	case MP_LAYOUT:
		vkDestroyPipelineLayout(device, model.layout,
			NULL /* allocator */);
	case MP_DESCRIPTOR_SET_LAYOUT:
		vkDestroyDescriptorSetLayout(device, model.dsetlayout,
			NULL /* allocator */);
	}
}

enum uistage {
	UI_PIPELINE_ALL,
	UP_SAMPLER,
	UP_ICON_VIEW,
	UP_IMAGE_MEMORY,
	UP_ICON_IMAGE,
	UP_BUFFER_MEMORY,
	UP_ICON_BUFFER,
	UP_INDEX_BUFFER,
	UP_VERT_BUFFER,
	UP_PIPELINE,
	UP_SHADERS,
	UP_FRAG_SHADER,
	UP_VERT_SHADER,
	UP_LAYOUT,
	UP_DESCRIPTOR_SET,
	UP_DESCRIPTOR_SET_LAYOUT,
};

void
cleanup_ui(enum uistage stage)
{
	switch (stage) {
	case UI_PIPELINE_ALL:
	case UP_SAMPLER:
		vkDestroySampler(device, ui.sampler, NULL /* allocator */);
	case UP_ICON_VIEW:
		for (size_t i = 0; i < ui.niconviews; i++)
			vkDestroyImageView(device, ui.icons[i],
				NULL /* allocator */);
	case UP_IMAGE_MEMORY:
		vkFreeMemory(device, ui.immem, NULL /* allocator */);
	case UP_ICON_IMAGE:
		for (size_t i = 0; i < ui.niconims; i++)
			vkDestroyImage(device, ui.iconims[i],
				NULL /* allocator */);
	case UP_BUFFER_MEMORY:
		vkFreeMemory(device, ui.bufmem, NULL /* allocator */);
	case UP_ICON_BUFFER:
		vkDestroyBuffer(device, ui.iconbuf, NULL /* allocator */);
	case UP_INDEX_BUFFER:
		vkDestroyBuffer(device, ui.idxbuf, NULL /* allocator */);
	case UP_VERT_BUFFER:
		vkDestroyBuffer(device, ui.vertbuf, NULL /* allocator */);
	case UP_PIPELINE:
		vkDestroyPipeline(device, ui.pipeline, NULL /* allocator */);
	case UP_SHADERS:
	case UP_FRAG_SHADER:
		vkDestroyShaderModule(device, ui.fragshader,
			NULL /* allocator */);
	case UP_VERT_SHADER:
		vkDestroyShaderModule(device, ui.vertshader,
			NULL /* allocator */);
	case UP_LAYOUT:
		vkDestroyPipelineLayout(device, ui.layout,
			NULL /* allocator */);
	case UP_DESCRIPTOR_SET:
		(void)NULL;
	case UP_DESCRIPTOR_SET_LAYOUT:
		vkDestroyDescriptorSetLayout(device, ui.dsetlayout,
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
		vkDestroyImageView(device, renderer.depth,
			NULL /* allocator */);
	case DEPTH_MEMORY:
		vkFreeMemory(device, renderer.memory, NULL /* allocator */);
	case DEPTH_BUFFER:
		vkDestroyImage(device, renderer.depthim,
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
	UI_PIPELINE,
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
		cleanup_ui(UI_PIPELINE_ALL);
	case UI_PIPELINE:
		cleanup_model(MODEL_PIPELINE_ALL);
	case MODEL_PIPELINE:
		cleanup_renderer(RENDERER_ALL);
	case RENDERER:
		cleanup_display(DISPLAY_ALL);
	case DISPLAY:
		break;
	}
}

static bool
read_shader(
	const char *filename,
	VkShaderModule *out,
	VkResult *res,
	const char **msg
) {
	*res = VK_SUCCESS;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		*msg = "couldn't open shader file for reading";
		return false;
	}
	if (fseek(f, 0, SEEK_END) < 0) {
		*msg = "couldn't seek to end of shader file";
		return false;
	}
	size_t codesz = ftell(f);
	if (fseek(f, 0, SEEK_SET) < 0) {
		*msg = "couldn't seek to start of shader file";
		return false;
	}
	uint32_t *code = malloc(codesz);
	if (code == NULL) {
		*msg = "couldn't allocate for shader code";
		return false;
	}
	/* TODO: is endianness an issue here? */
	size_t num = fread(code, 1, codesz, f);
	if (num != codesz) {
		if (ferror(f))
			*msg = "couldn't read shader code into buffer";
		else
			*msg = "unexpected end of file";
	}
	fclose(f);

	*res = vkCreateShaderModule(device, &(VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.codeSize = codesz,
		.pCode = code,
	}, NULL /* allocator */, out);
	if (*res != VK_SUCCESS) {
		*msg = "couldn't create shader module: ";
		return false;
	}

	return true;
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
	/* TODO: check that the mouse click is within model/ui bounds */
	switch (button) {
	case GLFW_MOUSE_BUTTON_LEFT:
		if (action == GLFW_RELEASE) {
			model.panning = false;
			model.rotating = false;
		}

		if ((action == GLFW_PRESS) && (mods & GLFW_MOD_SHIFT)) {
			double x, y;
			glfwGetCursorPos(win, &x, &y);
			model.lastmouse[0] = x;
			model.lastmouse[1] = y;
			model.panning = true;
		}
		if ((action == GLFW_PRESS) && (mods & GLFW_MOD_CONTROL)) {
			double x, y;
			glfwGetCursorPos(win, &x, &y);
			model.lastmouse[0] = x;
			model.lastmouse[1] = y;
			model.rotating = true;
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
	vec3_sub(shifted, model.camera, model.target);
	cart2sph(shiftedP, shifted);
	shiftedP[0] += y < 0 ? zoomspeed : -zoomspeed;
	sph2cart(shifted, shiftedP);
	vec3_add(model.camera, model.target, shifted);
}

static void
cursorcb(GLFWwindow *win, double x, double y)
{
	(void)win;
	if (model.panning) {
		vec2 dpos;
		vec2_sub(dpos, model.lastmouse, (vec2){ x, y });
		model.lastmouse[0] = x;
		model.lastmouse[1] = y;

		vec3 f, u, r;
		vec3_sub(f, model.target, model.camera);
		vec3_mul_cross(u, f, model.up);
		vec3_mul_cross(r, f, u);
		vec3_norm(u, u);
		vec3_norm(r, r);
		vec3_scale(u, u, panspeed * dpos[0]);
		vec3_scale(r, r, -panspeed * dpos[1]);

		vec3 diff = { 0, };
		vec3_add(diff, diff, u);
		vec3_add(diff, diff, r);
		vec3_add(model.camera, model.camera, diff);
		vec3_add(model.target, model.target, diff);
	}

	if (model.rotating) {
		vec2 dpos;
		vec2_sub(dpos, model.lastmouse, (vec2){ x, y });
		model.lastmouse[0] = x;
		model.lastmouse[1] = y;

		/* TODO: Fix the discontinuities */
		vec3 shifted, shiftedP;
		vec3_sub(shifted, model.camera, model.target);
		cart2sph(shiftedP, shifted);
		shiftedP[1] += model.up[1] * rotspeed * dpos[0];
		shiftedP[2] += model.up[1] * rotspeed * dpos[1];
		if (shiftedP[2] < -PI/2) {
			shiftedP[1] += PI;
			shiftedP[2] = -PI/2 + (-PI/2 - shiftedP[2]);
			model.up[1] *= -1;
		} else if (shiftedP[2] > PI/2) {
			shiftedP[1] += PI;
			shiftedP[2] = PI/2 - (shiftedP[2] - PI/2);
			model.up[1] *= -1;
		}
		if (shiftedP[1] < 0)
			shiftedP[1] += 2*PI;
		else if (shiftedP[1] > 2*PI)
			shiftedP[1] -= 2*PI;
		sph2cart(shifted, shiftedP);
		vec3_add(model.camera, model.target, shifted);
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

	model.nobjs = ARR_SZ(files);
	model.objs = nmalloc(sizeof(struct object), model.nobjs);
	if (model.objs == NULL) err(1, "couldn't allocate objects");
	model.bounds = nmalloc(sizeof(struct bounds), model.nobjs);
	if (model.bounds == NULL) err(1, "couldn't allocate bounding boxes");

	FILE *fs[ARR_SZ(files)];
	for (size_t i = 0; i < ARR_SZ(fs); i++) {
		fs[i] = fopen(files[i], "r");
		if (fs[i] == NULL)
			err(1, "couldn't open %s for reading", files[i]);
		if (fscanf(fs[i], "%zu\n", &model.objs[i].nprims) == EOF)
			err(1, "couldn't read nprims form %s", files[i]);
		model.objs[i].meshidx = model.meshcnt;
		model.meshcnt += 3*model.objs[i].nprims;
		/* TODO: alignment is determined by the GPU */
	}

	model.meshes = nmalloc(sizeof(struct modelvertex), model.meshcnt);
	if (model.meshes == NULL) err(1, "couldn't allocate meshes");
	for (size_t i = 0; i < ARR_SZ(fs); i++) {
		struct modelvertex *vi = model.meshes + model.objs[i].meshidx;
		struct boundingbox b = {
			.offset = { 0, 0, 0 },
			.extent = { -1, -1, -1 },
			.dir = { 1.0, 0.0, 0.0 },
		};
		for (size_t j = 0; j < model.objs[i].nprims; j++) {
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

		model.bounds[i].orig = b;
		vec4_scale(model.bounds[i].curr.offset, b.extent, -20);
		vec4_scale(model.bounds[i].curr.extent, b.extent, 40);
	}
	reset_camera();

	const char *icons[] = {
		"black.tga",
		"open.tga",
		"save.tga",
	};
	ui.nicons = ARR_SZ(icons);
	ui.nverts = ui.nicons * 4;
	ui.nidxs = ui.nicons * 6;
	if ((ui.verts = nmalloc(sizeof(struct uivertex), ui.nverts)) == NULL)
		err(1, "couldn't allocate vertices");
	if ((ui.idxs = nmalloc(sizeof(uint32_t), ui.nidxs)) == NULL)
		err(1, "couldn't allocate indices");
	if ((ui.texels = nmalloc(sizeof(unsigned char *), ui.nicons)) == NULL)
		err(1, "couldn't allocate texels");
	if ((ui.boxes = nmalloc(sizeof(vec2), ui.nicons)) == NULL)
		err(1, "couldn't allocate texel boxes");
	memcpy(ui.verts, (struct uivertex[]) {
		{ .pos = { 0, 0 } },
		{ .pos = { (1-meshpct)*width, 0 } },
		{ .pos = { 0, height } },
		{ .pos = { (1-meshpct)*width, height } },

		{ .pos = { 50, 120 } },
		{ .pos = { 50+32, 120 } },
		{ .pos = { 50, 120+32 } },
		{ .pos = { 50+32, 120+32 } },

		{ .pos = { 100, 120 } },
		{ .pos = { 100+32, 120 } },
		{ .pos = { 100, 120+32 } },
		{ .pos = { 100+32, 120+32 } },
	}, sizeof(struct uivertex) * ui.nverts);
	for (size_t i = 0; i < ui.nicons; i++) {
		size_t idx = i * 6;
		size_t vtx = i * 4;
		ui.idxs[idx+0] = vtx + 0;
		ui.idxs[idx+1] = vtx + 1;
		ui.idxs[idx+2] = vtx + 2;
		ui.idxs[idx+3] = vtx + 2;
		ui.idxs[idx+4] = vtx + 1;
		ui.idxs[idx+5] = vtx + 3;
	}

	ui.texelsz = 0;
	for (size_t i = 0; i < ui.nicons; i++) {
		int x, y, n, ok;
		ok = stbi_info(icons[i], &x, &y, &n);
		if (!ok) errx(1, "couldn't read size from %s", icons[i]);
		if (n != 4)
			errx(1, "%s doesn't have 4 colour channels", icons[i]);
		/* TODO: overflow */
		ui.texelsz += x * y * n;
		memcpy(ui.boxes[i], (vec2){ x, y }, sizeof(vec2));
	}
	if ((ui.texels = nmalloc(1, ui.texelsz)) == NULL)
		err(1, "couldn't allocate texel buffer");
	if ((ui.texoff = nmalloc(sizeof(size_t), ui.nicons)) == NULL)
		err(1, "couldn't allocate texture offset buffer");
	ui.texelsz = 0;
	for (size_t i = 0; i < ui.nicons; i++) {
		int x, y, n;
		unsigned char *data = stbi_load(icons[i], &x, &y, &n, 4);
		if (data == NULL)
			errx(1, "couldn't read %s", icons[i]);
		memcpy(ui.texels + ui.texelsz, data, x * y * n);
		ui.texoff[i] = ui.texelsz;
		ui.texelsz += x * y * n;
		stbi_image_free(data);
	}

	if ((ui.dsets = nmalloc(sizeof(VkDescriptorSet), ui.nicons)) == NULL)
		err(1, "couldn't allocate space for descriptor sets");
	if ((ui.iconoff = nmalloc(sizeof(size_t), ui.nicons)) == NULL)
		err(1, "couldn't allocate icon offset buffer");
	if ((ui.iconims = nmalloc(sizeof(VkImage), ui.nicons)) == NULL)
		err(1, "couldn't allocate space for images");
	if ((ui.imbinds = nmalloc(sizeof(VkBindImageMemoryInfo), ui.nicons))
			== NULL)
		err(1, "couldn't allocate space for image memory binds");
	if ((ui.icons = nmalloc(sizeof(VkImageView), ui.nicons)) == NULL)
		err(1, "couldn't allocate space for image views");
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
	const char **exts = nmalloc(sizeof(char **), nexts);
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
	VkPhysicalDevice *devs = nmalloc(sizeof(VkPhysicalDevice), devcnt);
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
	VkSurfaceFormat2KHR *fmts =
		nmalloc(sizeof(VkSurfaceFormat2KHR), fmtcap);
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
	VkPresentModeKHR *modes = nmalloc(sizeof(VkPresentModeKHR), modecap);
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
	VkImage *ims = nmalloc(sizeof(VkImage), imcap);
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
	display.semaphores = nmalloc(sizeof(VkSemaphore), imcnt);
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

	display.views = nmalloc(sizeof(VkImageView), display.nims);
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
setup_model(void)
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
	}, NULL /* allocator */, &model.dsetlayout);
	if (res != VK_SUCCESS) {
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate descriptor set layout: %s",
			vkstrerror(res));
	}

	res = vkAllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = renderer.dpool,
		.descriptorSetCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			model.dsetlayout,
		}
	}, &model.dset);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_DESCRIPTOR_SET_LAYOUT);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't get a descriptor set: %s", vkstrerror(res));
	}

	res = vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			model.dsetlayout,
		},
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL,
	}, NULL /* allocator */, &model.layout);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_DESCRIPTOR_SET_LAYOUT);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't create pipeline layout: %s", vkstrerror(res));
	}

	const char *msg;
	if (!read_shader("vert.model.spv", &model.vertshader, &res, &msg)) {
		cleanup_model(MP_LAYOUT);
		cleanup_before(MODEL_PIPELINE);
		if (res != VK_SUCCESS)
			errx(1, "%s: %s", msg, vkstrerror(res));
		else
			errx(1, "%s", msg);
	}
	if (!read_shader("frag.model.spv", &model.fragshader, &res, &msg)) {
		cleanup_model(MP_VERT_SHADER);
		cleanup_before(MODEL_PIPELINE);
		if (res != VK_SUCCESS)
			errx(1, "%s: %s", msg, vkstrerror(res));
		else
			errx(1, "%s", msg);
	}

	res = vkCreateGraphicsPipelines(device, NULL /* cache */, 1,
			&(VkGraphicsPipelineCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		/* TODO: */ .flags = 0,
		.stageCount = 2,
		.pStages = (VkPipelineShaderStageCreateInfo[]){ {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.pNext = NULL,
				.flags = 0,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = model.vertshader,
				.pName = "main",
				.pSpecializationInfo = NULL,
			}, {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 	.pNext = NULL,
			 	.flags = 0,
			 	.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			 	.module = model.fragshader,
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
				.stride = sizeof(struct modelvertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
			.vertexAttributeDescriptionCount = 2,
			.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){{
					.binding = 0,
					.location = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(struct modelvertex, pos),
				}, {
					.binding = 0,
					.location = 1,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(struct modelvertex, norm),
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
		.layout = model.layout,
		.renderPass = renderer.pass,
		.subpass = 0,
		.basePipelineHandle = 0,
		.basePipelineIndex = 0,
	}, NULL /* allocator */, &model.pipeline);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_SHADERS);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't create graphics pipeline: %s",
			vkstrerror(res));
	}

	/* TODO: put these bad bois in variables (size) */
	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(struct modelvertex) * model.meshcnt,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
		},
	}, NULL /* allocator */, &model.vertbuf);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_PIPELINE);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(mat4x4) * model.nobjs,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
		},
	}, NULL /* allocator */, &model.transbuf);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_VERTEX_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo){
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(mat4x4) * model.nobjs,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) {
			qfamidx,
		},
	}, NULL /* allocator */, &model.modelbuf);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_TRANSFORM_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	/* TODO: all of this can overflow */
	VkMemoryRequirements memreq;
	vkGetBufferMemoryRequirements(device, model.modelbuf, &memreq);
	size_t align = memreq.alignment - 1;
	model.meshsz =
		(sizeof(struct modelvertex) * model.meshcnt + align) & ~align;
	size_t transsz = (sizeof(mat4x4) * model.nobjs + align) & ~align;
	size_t modelsz = (sizeof(mat4x4) * model.nobjs + align) & ~align;
	size_t allocsz = max(memreq.size, model.meshsz + transsz + modelsz);
	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = allocsz,
		.memoryTypeIndex = memidx,
	}, NULL /* allocator */, &model.memory);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_MODEL_BUFFER);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't allocate device memory: %s", vkstrerror(res));
	}

	res = vkBindBufferMemory2(device, 3, (VkBindBufferMemoryInfo[]){{
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = model.vertbuf,
			.memory = model.memory,
			.memoryOffset = 0,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = model.transbuf,
			.memory = model.memory,
			.memoryOffset = model.meshsz,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = model.modelbuf,
			.memory = model.memory,
			.memoryOffset = model.meshsz + transsz,
		},
	});
	if (res != VK_SUCCESS) {
		cleanup_model(MP_DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't bind buffer memory: %s", vkstrerror(res));
	}

	/* TODO: investigate the consequences of this being one big mem block */
	void *ptr;
	res = vkMapMemory(device, model.memory, 0, model.meshsz, 0, &ptr); 
	if (res != VK_SUCCESS) {
		cleanup_model(MP_DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	memcpy(ptr, model.meshes, sizeof(struct modelvertex)*model.meshcnt);
	vkUnmapMemory(device, model.memory);

	res = vkMapMemory(device, model.memory, model.meshsz, transsz+modelsz,
		0, (void **)&model.transforms);
	if (res != VK_SUCCESS) {
		cleanup_model(MP_DEVICE_MEMORY);
		cleanup_before(MODEL_PIPELINE);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	model.models = (mat4x4 *)((char *)model.transforms + transsz);
}

static void
setup_ui(void)
{
	VkResult res;

	res = vkCreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = 1,
		.pBindings = (VkDescriptorSetLayoutBinding[]){ {
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				.pImmutableSamplers = NULL,
			},
		},
	}, NULL /* allocator */, &ui.dsetlayout);
	if (res != VK_SUCCESS) {
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate descriptor set layout: %s",
			vkstrerror(res));
	}

	VkDescriptorSetLayout *layouts =
		nmalloc(sizeof(VkDescriptorSetLayout), ui.nicons);
	if (layouts == NULL) {
		cleanup_ui(UP_DESCRIPTOR_SET_LAYOUT);
		cleanup_before(UI_PIPELINE);
		err(1, "couldn't allocate space for descriptor set layouts");
	}
	for (size_t i = 0; i < ui.nicons; i++)
		layouts[i] = ui.dsetlayout;
	res = vkAllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = renderer.dpool,
		.descriptorSetCount = ui.nicons,
		.pSetLayouts = layouts,
	}, ui.dsets);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_DESCRIPTOR_SET_LAYOUT);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't get a descriptor set: %s", vkstrerror(res));
	}

	res = vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = ui.nicons,
		.pSetLayouts = layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = (VkPushConstantRange[]){ {
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.offset = 0,
				.size = 3*sizeof(vec2),
			},
		},
	}, NULL /* allocator */, &ui.layout);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_DESCRIPTOR_SET);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't create pipeline layout: %s", vkstrerror(res));
	}
	free(layouts);

	const char *msg;
	if (!read_shader("vert.ui.spv", &ui.vertshader, &res, &msg)) {
		cleanup_ui(UP_LAYOUT);
		cleanup_before(UI_PIPELINE);
		if (res != VK_SUCCESS)
			errx(1, "%s: %s", msg, vkstrerror(res));
		else
			errx(1, "%s", msg);
	}
	if (!read_shader("frag.ui.spv", &ui.fragshader, &res, &msg)) {
		cleanup_ui(UP_VERT_SHADER);
		cleanup_before(UI_PIPELINE);
		if (res != VK_SUCCESS)
			errx(1, "%s: %s", msg, vkstrerror(res));
		else
			errx(1, "%s", msg);
	}

	res = vkCreateGraphicsPipelines(device, NULL /* cache */, 1,
			&(VkGraphicsPipelineCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		/* TODO: */ .flags = 0,
		.stageCount = 2,
		.pStages = (VkPipelineShaderStageCreateInfo[]){ {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.pNext = NULL,
				.flags = 0,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = ui.vertshader,
				.pName = "main",
				.pSpecializationInfo = NULL,
			}, {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 	.pNext = NULL,
			 	.flags = 0,
			 	.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			 	.module = ui.fragshader,
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
				.stride = sizeof(struct uivertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
			.vertexAttributeDescriptionCount = 1,
			.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){{
					.binding = 0,
					.location = 0,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = offsetof(struct uivertex, pos),
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
				.x = 0,
				.y = 0,
				.width = (1-meshpct) * width,
				.height = height,
				.minDepth = 0.0,
				.maxDepth = 1.0,
			},
			.scissorCount = 1,
			.pScissors = &(VkRect2D){
				.offset = (VkOffset2D){
					.x = 0,
					.y = 0
				},
				.extent = (VkExtent2D){
					.width = (1-meshpct) * width,
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
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
			.depthTestEnable = VK_FALSE,
			.depthWriteEnable = VK_FALSE,
			.depthCompareOp = VK_COMPARE_OP_ALWAYS,
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
		.layout = ui.layout,
		.renderPass = renderer.pass,
		.subpass = 1,
		.basePipelineHandle = 0,
		.basePipelineIndex = 0,
	}, NULL /* allocator */, &ui.pipeline);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_SHADERS);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't create graphics pipeline: %s",
			vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(struct uivertex) * ui.nverts,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) { qfamidx },
	}, NULL /* allocator */, &ui.vertbuf);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_PIPELINE);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = sizeof(uint32_t) * ui.nidxs,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) { qfamidx },
	}, NULL /* allocator */, &ui.idxbuf);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_VERT_BUFFER);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	res = vkCreateBuffer(device, &(VkBufferCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = ui.texelsz,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = (uint32_t[]) { qfamidx },
	}, NULL /* allocator */, &ui.iconbuf);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_INDEX_BUFFER);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	/* TODO: overflow */
	/* TODO: alignment could be different per buffer / image */
	VkMemoryRequirements memreq;
	vkGetBufferMemoryRequirements(device, ui.vertbuf, &memreq);
	size_t align = memreq.alignment - 1;
	size_t vertsz = (sizeof(struct uivertex) * ui.nverts + align) & ~align;
	size_t idxsz = (sizeof(uint32_t) * ui.nidxs + align) & ~align;
	size_t texelsz = (ui.texelsz + align) & ~align;
	size_t allocsz = vertsz + idxsz + texelsz;
	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = allocsz,
		.memoryTypeIndex = memidx,
	}, NULL /* allocator */, &ui.bufmem);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_INDEX_BUFFER);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate device memory: %s", vkstrerror(res));
	}

	res = vkBindBufferMemory2(device, 3, (VkBindBufferMemoryInfo[]){{
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = ui.vertbuf,
			.memory = ui.bufmem,
			.memoryOffset = 0,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = ui.idxbuf,
			.memory = ui.bufmem,
			.memoryOffset = vertsz,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = ui.iconbuf,
			.memory = ui.bufmem,
			.memoryOffset = vertsz + idxsz,
		},
	});
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_BUFFER_MEMORY);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't bind buffer memory: %s", vkstrerror(res));
	}

	unsigned char *ptr;
	res = vkMapMemory(device, ui.bufmem, 0, allocsz, 0, (void **)&ptr); 
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_BUFFER_MEMORY);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	size_t off = 0;
	memcpy(ptr+off, ui.verts, sizeof(struct uivertex) * ui.nverts);
	off += vertsz;
	memcpy(ptr+off, ui.idxs, sizeof(uint32_t) * ui.nidxs);
	off += idxsz;
	memcpy(ptr+off, ui.texels, ui.texelsz);
	vkUnmapMemory(device, ui.bufmem);

	ui.niconims = 0;
	allocsz = 0;
	for (size_t i = 0; i < ui.nicons; i++) {
		res = vkCreateImage(device, &(VkImageCreateInfo){
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_R8G8B8A8_UINT,
			.extent = {
				.width = ui.boxes[i][0],
				.height = ui.boxes[i][1],
				.depth = 1,
			},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
				VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = (uint32_t[]){ qfamidx },
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		}, NULL /* allocator */, ui.iconims+i);
		if (res != VK_SUCCESS) {
			cleanup_ui(UP_ICON_IMAGE);
			cleanup_before(RENDERER);
			errx(1, "couldn't create image for depth buffer: %s",
				vkstrerror(res));
		}
		ui.niconims++;

		VkMemoryRequirements2 imreqs = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
			.pNext = NULL,
		};
		vkGetImageMemoryRequirements2(device, &(VkImageMemoryRequirementsInfo2){
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
			.pNext = NULL,
			.image = ui.iconims[i],
		}, &imreqs);
		align = imreqs.memoryRequirements.alignment;
		allocsz = (allocsz + align) & ~align;
		ui.iconoff[i] = allocsz;
		allocsz += imreqs.memoryRequirements.size;
	}

	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = allocsz,
		.memoryTypeIndex = memidx,
	}, NULL /* allocator */, &ui.immem);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_INDEX_BUFFER);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't allocate device memory: %s", vkstrerror(res));
	}

	for (size_t i = 0; i < ui.nicons; i++)
		ui.imbinds[i] = (VkBindImageMemoryInfo){
			.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
			.pNext = NULL,
			.image = ui.iconims[i],
			.memory = ui.immem,
			.memoryOffset = ui.iconoff[i],
		};
	res = vkBindImageMemory2(device, ui.nicons, ui.imbinds);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_IMAGE_MEMORY);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't bind image memory: %s", vkstrerror(res));
	}


	ui.niconviews = 0;
	for (size_t i = 0; i < ui.nicons; i++) {
		res = vkCreateImageView(device, &(VkImageViewCreateInfo){
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.image = ui.iconims[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_R8G8B8A8_UINT,
			.components = { 0, },
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}
		}, NULL /* allocator */, ui.icons+i);
		if (res != VK_SUCCESS) {
			cleanup_ui(UP_ICON_VIEW);
			cleanup_before(UI_PIPELINE);
			errx(1, "couldn't create image view for icon: %s",
				vkstrerror(res));
		}
		ui.niconviews++;
	}

	res = vkCreateSampler(device, &(VkSamplerCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.mipLodBias = 0,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 0,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_NEVER,
		.minLod = 0,
		.maxLod = 0,
		.unnormalizedCoordinates = VK_FALSE,
	}, NULL /* allocator */, &ui.sampler);
	if (res != VK_SUCCESS) {
		cleanup_ui(UP_ICON_VIEW);
		cleanup_before(UI_PIPELINE);
		errx(1, "couldn't create image sampler: %s", vkstrerror(res));
	}
}

static void
setup_renderer(void)
{
	VkResult res;

	res = vkCreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.maxSets = 1 + ui.nicons,
		.poolSizeCount = 3,
		.pPoolSizes = (VkDescriptorPoolSize[]){ {
				.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
			}, {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 2,
			}, {
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = ui.nicons,
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
	}, NULL /* allocator */, &renderer.depthim);
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
		.image = renderer.depthim,
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
			.image = renderer.depthim,
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
		.image = renderer.depthim,
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
	}, NULL /* allocator */, &renderer.depth);
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
		.subpassCount = 2,
		.pSubpasses = (VkSubpassDescription[]){ {
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
			}, {
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
		},
		.dependencyCount = 2,
		.pDependencies = (VkSubpassDependency[]){ {
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			}, {
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 1,
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

	display.framebuffers = nmalloc(sizeof(VkFramebuffer), display.nims);
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
				renderer.depth,
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
prerender_model(void)
{
	for (size_t i = 0; i < model.nobjs; i++) {
		mat4x4 A, M, V, P;

		const float *off = model.bounds[i].orig.offset;
		const float *box = model.bounds[i].curr.extent;
		const float *ogbox = model.bounds[i].orig.extent;
		mat4x4_identity(A);
		mat4x4_scale_aniso(M, A,
			box[0] / ogbox[0],
			box[1] / ogbox[1],
			box[2] / ogbox[2]);
		mat4x4_translate(A, off[0], off[1], off[2]);
		mat4x4_mul(M, A, M);
		mat4x4_look_at(V, model.camera, model.target, model.up);
		mat4x4_perspective(P, 120, width/height, -1, 1);
		mat4x4_mul(A, V, M);
		mat4x4_mul(model.transforms[i], P, A);
		mat4x4_dup(model.models[i], M);
	}
	vkUpdateDescriptorSets(device, 1, (VkWriteDescriptorSet[]){{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = NULL,
			.dstSet = model.dset,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pImageInfo = NULL,
			.pBufferInfo = (VkDescriptorBufferInfo[]){{
					.buffer = model.transbuf,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				}, {
					.buffer = model.modelbuf,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				},
			},
			.pTexelBufferView = NULL,
		},
	}, 0, NULL);
}

static void
render_model(void)
{
	vkCmdBindDescriptorSets(renderer.cmdbuf,
		VK_PIPELINE_BIND_POINT_GRAPHICS, model.layout, 0, 1,
		&model.dset, 2, (uint32_t[]){ 0, 0 });
	vkCmdBindPipeline(renderer.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
		model.pipeline);
	vkCmdBindVertexBuffers(renderer.cmdbuf, 0, 1,
		(VkBuffer[]){ model.vertbuf },
		(VkDeviceSize[]){ 0 });
	vkCmdDraw(renderer.cmdbuf, model.meshcnt, 1, 0, 0);
}

static void
prerender_ui(void)
{

	for (size_t i = 0; i < ui.nicons; i++) {
		vkCmdPipelineBarrier(renderer.cmdbuf,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_DEPENDENCY_BY_REGION_BIT,
			0, NULL,
			0, NULL,
			1, (VkImageMemoryBarrier[]){ {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.pNext = NULL,
					.srcAccessMask = 0,
					.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = ui.iconims[i],
					.subresourceRange = (VkImageSubresourceRange){
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					},
				},
			});
		vkCmdCopyBufferToImage(renderer.cmdbuf, ui.iconbuf,
			ui.iconims[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &(VkBufferImageCopy){
				.bufferOffset = ui.texoff[i],
				.bufferRowLength = ui.boxes[i][0],
				.bufferImageHeight = ui.boxes[i][1],
				.imageSubresource = (VkImageSubresourceLayers){
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.imageOffset = (VkOffset3D){
					.x = 0,
					.y = 0,
					.z = 0,
				},
				.imageExtent = (VkExtent3D){
					.width = ui.boxes[i][0],
					.height = ui.boxes[i][1],
					.depth = 1,
				},
			});
		vkCmdPipelineBarrier(renderer.cmdbuf,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_DEPENDENCY_BY_REGION_BIT,
			0, NULL,
			0, NULL,
			1, (VkImageMemoryBarrier[]){ {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.pNext = NULL,
					.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
					.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = ui.iconims[i],
					.subresourceRange = (VkImageSubresourceRange){
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					},
				},
			});
		vkUpdateDescriptorSets(device, 1, (VkWriteDescriptorSet[]){{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.pNext = NULL,
				.dstSet = ui.dsets[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &(VkDescriptorImageInfo) {
					.sampler = ui.sampler,
					.imageView = ui.icons[i],
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				},
				.pBufferInfo = NULL,
				.pTexelBufferView = NULL,
			},
		}, 0, NULL);
	}
}

static void
render_ui(void)
{
	vkCmdBindPipeline(renderer.cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
		ui.pipeline);
	vkCmdBindVertexBuffers(renderer.cmdbuf, 0, 1,
		(VkBuffer[]){ ui.vertbuf }, (VkDeviceSize[]){ 0 });
	vkCmdBindIndexBuffer(renderer.cmdbuf, ui.idxbuf, 0,
		VK_INDEX_TYPE_UINT32);
	vkCmdPushConstants(renderer.cmdbuf, ui.layout,
		VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vec2),
		&(vec2){ (1-meshpct)*width, height });
	for (size_t i = 0; i < ui.nicons; i++) {
		vkCmdBindDescriptorSets(renderer.cmdbuf,
			VK_PIPELINE_BIND_POINT_GRAPHICS, ui.layout, 0, 1,
			ui.dsets+i, 0, NULL);
		vkCmdPushConstants(renderer.cmdbuf, ui.layout,
			VK_SHADER_STAGE_VERTEX_BIT, sizeof(vec2), sizeof(vec2),
			ui.boxes[i]);
		vkCmdPushConstants(renderer.cmdbuf, ui.layout,
			VK_SHADER_STAGE_VERTEX_BIT, 2*sizeof(vec2),
			sizeof(vec2), ui.verts[6*i].pos);
		vkCmdDrawIndexed(renderer.cmdbuf, 6, 1, i*6, 0, 0);
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
		return;
	}

	prerender_model();
	prerender_ui();
	vkCmdBeginRenderPass(renderer.cmdbuf, &(VkRenderPassBeginInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = renderer.pass,
		.framebuffer = display.framebuffers[idx],
		.renderArea = {
			.offset = { .x = 0, .y = 0 },
			.extent = { .width = width, .height = height },
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
	render_model();
	vkCmdNextSubpass(renderer.cmdbuf, VK_SUBPASS_CONTENTS_INLINE);
	render_ui();
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
		.pCommandBuffers = (VkCommandBuffer[]){ renderer.cmdbuf },
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
}

int
main(void)
{
	setup_cpu();
	setup_display();
	setup_renderer();
	setup_model();
	setup_ui();
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
