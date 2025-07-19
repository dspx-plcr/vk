#include <err.h>
#include <inttypes.h>
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

struct display {
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
	
};

static const char *progname = "My Little Vulkan App";
static const float width = 1920;
static const float height = 1080;
static const float depth = 1000;

static VkInstance instance;
static VkDevice device;
static uint32_t qfamidx;
static struct display display;
static VkRenderPass renderpass;
static VkDescriptorSetLayout dsetlayout;
static VkDescriptorPool dpool;
static VkPipelineLayout layout;
static VkShaderModule vertshader;
static VkShaderModule fragshader;
static VkPipeline pipeline;
static VkCommandPool pool;
static VkDeviceMemory memory;
static VkBuffer vertbuf;
static VkBuffer transbuf;
static VkFence fence;
static VkSemaphore acquiresem;
static VkCommandBuffer cmdbuf;

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
static mat4x4 *transforms;

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

static char
usabledev(VkPhysicalDevice dev, VkPhysicalDeviceProperties2 props)
{
	if (VK_API_VERSION_MAJOR(props.properties.apiVersion) != 1 ||
			VK_API_VERSION_MINOR(props.properties.apiVersion) < 1)
		return 0;

	if (props.properties.limits.maxViewportDimensions[0] < width ||
			props.properties.limits.maxViewportDimensions[1] < height)
		return 0;

	uint32_t qcnt;
	vkGetPhysicalDeviceQueueFamilyProperties2(dev, &qcnt, NULL);
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
	char found = 0;
	for (size_t i = 0; i < qcnt; i++) {
		VkQueueFlags flags = qprops[i].queueFamilyProperties.queueFlags;
		if (!(flags & VK_QUEUE_GRAPHICS_BIT))
			continue;
		if (glfwGetPhysicalDevicePresentationSupport(
				instance, dev, i)) {
			found = 1;
			qfamidx = i;
			break;
		}
	}
	free(qprops);
	return found;
}

static char
preferdev(VkPhysicalDeviceProperties2 new, VkPhysicalDeviceProperties2 old)
{
	if (old.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
		return 1;
	if (new.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			old.properties.deviceType !=
			VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		return 1;

	return 0;
}

enum stage {
	INSTANCE,
	SURFACE,
	DEVICE,
	SWAP_CHAIN,
	SEMAPHORES,
	IMAGE_VIEWS,
	RENDER_PASS,
	FRAMEBUFFERS,
	DESCRIPTOR_SET_LAYOUT,
	LAYOUT,
	VERT_SHADER,
	FRAG_SHADER,
	SHADERS,
	PIPELINE,
	POOL,
	DESCRIPTOR_POOL,
	VERTEX_BUFFER,
	TRANSFORM_BUFFER,
	DEVICE_MEMORY,
	UNMAP_TRANSFORMS,
	FENCE,
	ACQUIRE_SEMAPHORE,
	COMMAND_BUFFER,
	ALL,
};

static void
cleanupgpu(enum stage s)
{
	switch (s) {
		VkQueue queue;
	case ALL:
		vkGetDeviceQueue(device, qfamidx, 0, &queue);
		vkQueueWaitIdle(queue);
		vkDeviceWaitIdle(device);
	case COMMAND_BUFFER:
		vkFreeCommandBuffers(device, pool, 1, &cmdbuf);
	case ACQUIRE_SEMAPHORE:
		vkDestroySemaphore(device, acquiresem, NULL /* allocator */);
	case FENCE:
		vkDestroyFence(device, fence, NULL /* allocator */);
	case UNMAP_TRANSFORMS:
		vkUnmapMemory(device, memory);
	case DEVICE_MEMORY:
		vkFreeMemory(device, memory, NULL /* allocator */);
	case TRANSFORM_BUFFER:
		vkDestroyBuffer(device, transbuf, NULL /* allocator */);
	case VERTEX_BUFFER:
		vkDestroyBuffer(device, vertbuf, NULL /* allocator */);
	case DESCRIPTOR_POOL:
		vkDestroyDescriptorPool(device, dpool, NULL /* allocator */);
	case POOL:
		vkDestroyCommandPool(device, pool, NULL /* allocator */);
	case PIPELINE:
		vkDestroyPipeline(device, pipeline, NULL /* allocator */);
	case SHADERS:
	case FRAG_SHADER:
		vkDestroyShaderModule(device, fragshader, NULL /* allocator */);
	case VERT_SHADER:
		vkDestroyShaderModule(device, vertshader, NULL /* allocator */);
	case LAYOUT:
		vkDestroyPipelineLayout(device, layout, NULL /* allocator */);
	case DESCRIPTOR_SET_LAYOUT:
		vkDestroyDescriptorSetLayout(
			device, dsetlayout, NULL /* allocator */);
	case FRAMEBUFFERS:
		for (size_t i = 0; i < display.nfbs; i++)
			vkDestroyFramebuffer(device, display.framebuffers[i],
				NULL /* allocator */);
		display.nfbs = 0;
	case RENDER_PASS:
		vkDestroyRenderPass(device, renderpass, NULL /* allocator */);
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

static VkShaderModule
readshader(const char *filename, enum stage toclean)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		cleanupgpu(toclean);
		err(1, "couldn't open shader file for reading");
	}
	if (fseek(f, 0, SEEK_END) < 0) {
		cleanupgpu(toclean);
		err(1, "couldn't seek to end of shader file");
	}
	size_t codesz = ftell(f);
	if (fseek(f, 0, SEEK_SET) < 0) {
		cleanupgpu(toclean);
		err(1, "couldn't seek to start of shader file");
	}
	uint32_t *code = malloc(codesz);
	if (code == NULL) {
		cleanupgpu(toclean);
		err(1, "couldn't allocate for shader code");
	}
	/* TODO: is endianness an issue here? */
	size_t num = fread(code, 1, codesz, f);
	if (num != codesz) {
		cleanupgpu(toclean);
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
		cleanupgpu(toclean);
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
}

static void
expandbox(struct boundingbox *box, vec3 point, size_t axis)
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
setupcpu(void)
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
				if (res == EOF)
					err(1, "couldn't read data from %s",
						files[i]);
				expandbox(&b, vert, 0);
				expandbox(&b, vert, 1);
				expandbox(&b, vert, 2);
			}
			if (fscanf(fs[i], "\n") == EOF)
				err(1, "couldn't read data from %s", files[i]);
		}
		fclose(fs[i]);

		bounds[i].orig = b;
		vec4_scale(bounds[i].curr.offset, b.extent, -20);
		vec4_scale(bounds[i].curr.extent, b.extent, 40);
	}

	camera[0] = 0;
	camera[1] = height/10;
	camera[2] = depth/4;
}

static void
setupgpu(void)
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
	/* TODO: technically can overflow */
	const char **exts = malloc(
		sizeof(char **) * (numglfwexts + ARR_SZ(manualexts)));
	if (exts == NULL)
		err(1, "couldn't allocate for instance extensions");
	memcpy(exts, glfwexts, sizeof(char *)*numglfwexts);
	memcpy(exts+numglfwexts, manualexts, sizeof(manualexts));
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
		cleanupgpu(INSTANCE);
		errx(1, "couldn't create a GLFW window: %s", msg);
	}

	res = glfwCreateWindowSurface(instance, display.window,
		NULL /* allocator */, &display.surface);
	if (res != VK_SUCCESS) {
		cleanupgpu(INSTANCE);
		errx(1, "couldn't create surface for window: %s",
			vkstrerror(res));
	}
	glfwSetKeyCallback(display.window, &keycb);

	uint32_t devcnt;
	res = vkEnumeratePhysicalDevices(instance, &devcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanupgpu(INSTANCE);
		errx(1, "coudln't get number of physical devices: %s",
			vkstrerror(res));
	}
	VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * devcnt);
	if (devs == NULL) {
		cleanupgpu(INSTANCE);
		err(1, "couldn't allocate to store physical devices");
	}
	size_t devcap = devcnt;
	while ((res = vkEnumeratePhysicalDevices(instance, &devcnt, devs))
			== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkPhysicalDevice) > devcap) {
			cleanupgpu(INSTANCE);
			errx(1, "can't store %zu physical devices", devcap);
		}
		devcap *= 2;
		VkPhysicalDevice *tmp = realloc(
			devs, sizeof(VkPhysicalDevice) * devcap);
		if (tmp == NULL) {
			cleanupgpu(INSTANCE);
			err(1, "coudln't allocate to store physical devices");
		}
		devs = tmp;
		devcnt = devcap;
	}
	if (res != VK_SUCCESS) {
		cleanupgpu(INSTANCE);
		errx(1, "couldn't get physical devices: %s", vkstrerror(res));
	}

	VkPhysicalDevice physdev;
	VkPhysicalDeviceProperties2 devprops;
	char found = 0;
	for (size_t i = 0; i < devcnt; i++) {
		VkPhysicalDeviceProperties2 props = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = NULL,
		};
		vkGetPhysicalDeviceProperties2(devs[i], &props);
		if (!usabledev(devs[i], props))
			continue;

		if (!found) {
			physdev = devs[i];
			devprops = props;
			found = 1;
			continue;
		}

		if (preferdev(props, devprops)) {
			physdev = devs[i];
			devprops = props;
		}
	}
	free(devs);

	if (!found) {
		cleanupgpu(INSTANCE);
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
		cleanupgpu(INSTANCE);
		errx(1, "couldn't create logical device: %s", vkstrerror(res));
	}

	VkSurfaceCapabilitiesKHR caps;
	res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
		physdev, display.surface, &caps);
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE);
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
		cleanupgpu(DEVICE);
		errx(1, "couldn't query for surface formats: %s",
			vkstrerror(res));
	}
	size_t fmtcap = fmtcnt;
	/* TODO: can overflow */
	VkSurfaceFormat2KHR *fmts = malloc(sizeof(VkSurfaceFormat2KHR)*fmtcap);
	if (fmts == NULL) {
		cleanupgpu(DEVICE);
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
			cleanupgpu(DEVICE);
			errx(1, "can't store %zu physical devices", fmtcap);
		}
		fmtcap *= 2;
		VkSurfaceFormat2KHR *tmp = realloc(
			fmts, sizeof(VkSurfaceFormat2KHR) * fmtcap);
		if (tmp == NULL) {
			cleanupgpu(DEVICE);
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
		cleanupgpu(DEVICE);
		errx(1, "couldn't get surface formats: %s", vkstrerror(res));
	}

	uint32_t modecnt;
	res = vkGetPhysicalDeviceSurfacePresentModesKHR(
		physdev, display.surface, &modecnt, NULL);
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE);
		errx(1, "couldn't query for surface formats: %s",
			vkstrerror(res));
	}
	size_t modecap = modecnt;
	/* TODO: can overflow */
	VkPresentModeKHR *modes = malloc(sizeof(VkPresentModeKHR)*modecap);
	if (modes == NULL) {
		cleanupgpu(DEVICE);
		err(1, "couldn't allocate for surface formats");
	}
	while ((res = vkGetPhysicalDeviceSurfacePresentModesKHR(
			physdev, display.surface, &modecnt, modes))
	       		== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkPresentModeKHR) > modecap) {
			cleanupgpu(DEVICE);
			errx(1, "can't store %zu physical devices", modecap);
		}
		modecap *= 2;
		VkPresentModeKHR *tmp = realloc(
			modes, sizeof(VkPresentModeKHR) * modecap);
		if (tmp == NULL) {
			cleanupgpu(DEVICE);
			err(1, "coudln't allocate to store surface formats");
		}
		modes = tmp;
		modecnt = modecap;
	}
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE);
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
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
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
		cleanupgpu(DEVICE);
		errx(1, "couldn't create swapchain: %s", vkstrerror(res));
	}

	uint32_t imcnt;
	res = vkGetSwapchainImagesKHR(device, display.swapchain, &imcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanupgpu(SWAP_CHAIN);
		errx(1, "couldn't get swapchain images: %s", vkstrerror(res));
	}
	size_t imcap = imcnt;
	/* TODO: can overflow */
	VkImage *ims = malloc(sizeof(VkImage) * imcap);
	if (ims == NULL) {
		cleanupgpu(SWAP_CHAIN);
		err(1, "couldn't allocate memory for images");
	}
	while ((res = vkGetSwapchainImagesKHR(
			device, display.swapchain, &imcnt, ims))
	       		== VK_INCOMPLETE) {
		if (SIZE_MAX / 2 / sizeof(VkImage) > imcap) {
			cleanupgpu(SWAP_CHAIN);
			errx(1, "can't store %zu physical devices", imcap);
		}
		imcap *= 2;
		VkImage *tmp = realloc(modes, sizeof(VkImage) * imcap);
		if (tmp == NULL) {
			cleanupgpu(SWAP_CHAIN);
			err(1, "coudln't allocate to store surface formats");
		}
		ims = tmp;
		imcnt = imcap;
	}
	display.images = ims;
	display.nims = imcnt;

	display.semaphores = malloc(sizeof(VkSemaphore) * imcnt);
	if (display.semaphores == NULL) {
		cleanupgpu(SWAP_CHAIN);
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
			cleanupgpu(SEMAPHORES);
			errx(1, "couldn't create semaphore: %s", vkstrerror(res));
		}
		display.nsems++;
	}

	display.views = malloc(sizeof(VkImageView) * display.nims);
	if (display.views == NULL) {
		cleanupgpu(SEMAPHORES);
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
			.components =  {
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
			cleanupgpu(IMAGE_VIEWS);
			errx(1, "couldn't create image view: %s", vkstrerror(res));
		}
		display.nviews++;
	}

	res = vkCreateRenderPass(device, &(VkRenderPassCreateInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = 1,
		.pAttachments = (const VkAttachmentDescription[]){ {
				.flags = 0,
				.format = display.fmt.format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL,
		},
		.dependencyCount = 0,
		.pDependencies = NULL,
	}, NULL /* allocator */, &renderpass);
	if (res != VK_SUCCESS) {
		cleanupgpu(IMAGE_VIEWS);
		errx(1, "coudln't create render pass: %s", vkstrerror(res));
	}

	display.framebuffers = malloc(sizeof(VkFramebuffer) * imcnt);
	if (display.framebuffers == NULL) {
		cleanupgpu(IMAGE_VIEWS);
		err(1, "couldn't not allocate for framebuffers");
	}
	display.nfbs = 0;
	for (size_t i = 0; i < imcnt; i++) {
		res = vkCreateFramebuffer(device, &(VkFramebufferCreateInfo){
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.renderPass = renderpass,
			.attachmentCount = 1,
			.pAttachments = (VkImageView[]) {
				display.views[i],
			},
			.width = width,
			.height = height,
			.layers = 1,
		}, NULL /* allocator */, display.framebuffers+i);
		if (res != VK_SUCCESS) {
			cleanupgpu(FRAMEBUFFERS);
			errx(1, "coudln't allocate frame buffer: %s",
				vkstrerror(res));
		}
		display.nfbs++;
	}

	res = vkCreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = 1,
		.pBindings = (VkDescriptorSetLayoutBinding[]){ {
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.pImmutableSamplers = NULL,
			},
		},
	}, NULL /* allocator */, &dsetlayout);
	if (res != VK_SUCCESS) {
		cleanupgpu(FRAMEBUFFERS);
		errx(1, "couldn't allocate descriptor set layout: %s",
			vkstrerror(res));
	}

	res = vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			dsetlayout,
		},
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL,
	}, NULL /* allocator */, &layout);
	if (res != VK_SUCCESS) {
		cleanupgpu(DESCRIPTOR_SET_LAYOUT);
		errx(1, "couldn't create pipeline layout: %s", vkstrerror(res));
	}

	vertshader = readshader("vert.spv", LAYOUT);
	fragshader = readshader("frag.spv", VERT_SHADER);

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
				.module = vertshader,
				.pName = "main",
				.pSpecializationInfo = NULL,
			}, {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 	.pNext = NULL,
			 	.flags = 0,
			 	.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			 	.module = fragshader,
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
				.x = 0,
				.y = 0,
				.width = width,
				.height = height,
				.minDepth = 0.0,
				.maxDepth = 1.0,
			},
			.scissorCount = 1,
			.pScissors = &(VkRect2D){
				.offset = (VkOffset2D){ .x = 0, .y = 0 },
				.extent = (VkExtent2D){
					.width = width,
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
			.cullMode = VK_CULL_MODE_BACK_BIT,
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
			.depthCompareOp = 0,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
			.front = { 0 },
			.back = { 0 },
			.minDepthBounds = 0,
			.maxDepthBounds = 0,
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
		.layout = layout,
		.renderPass = renderpass,
		.subpass = 0,
		.basePipelineHandle = 0,
		.basePipelineIndex = 0,
	}, NULL /* allocator */, &pipeline);
	if (res != VK_SUCCESS) {
		cleanupgpu(SHADERS);
		errx(1, "couldn't create graphics pipeline: %s",
			vkstrerror(res));
	}

	res = vkCreateCommandPool(device, &(VkCommandPoolCreateInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = qfamidx,
	}, NULL /* allocator */, &pool);
	if (res != VK_SUCCESS) {
		cleanupgpu(PIPELINE);
		errx(1, "couldn't allocate command pool: %s", vkstrerror(res));
	}

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
				.descriptorCount = 1,
			},
		},
	}, NULL /* allocator */, &dpool);
	if (res != VK_SUCCESS) {
		cleanupgpu(POOL);
		errx(1, "coudln't create descriptor pool: %s", vkstrerror(res));
	}

	/* TODO: Do this during usabledev */
	VkPhysicalDeviceMemoryProperties2 memprops = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		.pNext = NULL,
	};
	vkGetPhysicalDeviceMemoryProperties2(physdev, &memprops);
	int64_t memidx = -1;
	for (size_t i = 0; i < memprops.memoryProperties.memoryTypeCount; i++) {
		VkMemoryPropertyFlags flags =
			memprops.memoryProperties.memoryTypes[i].propertyFlags;
		if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
				(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
			memidx = i;
			break;
		}
	}
	if (memidx < 0) {
		cleanupgpu(DESCRIPTOR_POOL);
		errx(1, "couldn't find usable device memory");
	}

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
	}, NULL /* allocator */, &vertbuf);
	if (res != VK_SUCCESS) {
		cleanupgpu(DESCRIPTOR_POOL);
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
	}, NULL /* allocator */, &transbuf);
	if (res != VK_SUCCESS) {
		cleanupgpu(VERTEX_BUFFER);
		errx(1, "couldn't allocate a buffer: %s", vkstrerror(res));
	}

	VkMemoryRequirements memreq;
	vkGetBufferMemoryRequirements(device, vertbuf, &memreq);
	size_t align = memreq.alignment - 1;
	meshsz = (sizeof(struct vertexinfo)*meshcnt + align) & ~align;
	size_t allocsz = max(memreq.size, meshsz + sizeof(mat4x4) * nobjs);
	allocsz = (allocsz + align) & ~align;
	res = vkAllocateMemory(device, &(VkMemoryAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = allocsz,
		.memoryTypeIndex = (uint32_t)memidx,
	}, NULL /* allocator */, &memory);
	if (res != VK_SUCCESS) {
		cleanupgpu(TRANSFORM_BUFFER);
		errx(1, "couldn't allocate device memory: %s", vkstrerror(res));
	}

	res = vkBindBufferMemory2(device, 2, (VkBindBufferMemoryInfo[]){{
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = vertbuf,
			.memory = memory,
			.memoryOffset = 0,
		}, {
			.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
			.pNext = NULL,
			.buffer = transbuf,
			.memory = memory,
			.memoryOffset = meshsz,
		},
	});
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE_MEMORY);
		errx(1, "couldn't bind buffer memory: %s", vkstrerror(res));
	}

	void *ptr;
	res = vkMapMemory(device, memory, 0, meshsz, 0, &ptr); 
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE_MEMORY);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}
	memcpy(ptr, meshes, sizeof(struct vertexinfo)*meshcnt);
	vkUnmapMemory(device, memory);

	res = vkMapMemory(device, memory, meshsz, allocsz-meshsz, 0,
		(void **)&transforms);
	if (res != VK_SUCCESS) {
		cleanupgpu(DEVICE_MEMORY);
		errx(1, "couldn't map buffer memory: %s", vkstrerror(res));
	}

	res = vkCreateFence(device, &(VkFenceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
	}, NULL /* allocator */, &fence);
	if (res != VK_SUCCESS) {
		cleanupgpu(UNMAP_TRANSFORMS);
		errx(1, "couldn't create fence: %s", vkstrerror(res));
	}

	res = vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
	}, NULL /* allocator */, &acquiresem);
	if (res != VK_SUCCESS) {
		cleanupgpu(FENCE);
		errx(1, "couldn't create semaphore: %s", vkstrerror(res));
	}

	res = vkAllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	}, &cmdbuf);
	if (res != VK_SUCCESS) {
		cleanupgpu(ACQUIRE_SEMAPHORE);
		errx(1, "couldn't get a command buffer: %s", vkstrerror(res));
	}
}

static void
teardown(void)
{
	cleanupgpu(ALL);
	glfwDestroyWindow(display.window);
	glfwTerminate();
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
		.semaphore = acquiresem,
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
		.descriptorPool = dpool,
		.descriptorSetCount = 1,
		.pSetLayouts = (VkDescriptorSetLayout[]){
			dsetlayout,
		}
	}, &dset);
	if (res != VK_SUCCESS) {
		warnx("couldn't get a descriptor set: %s", vkstrerror(res));
		return;
	}

	res = vkResetCommandBuffer(cmdbuf, 0);
	if (res != VK_SUCCESS) {
		warnx("couldn't reset command buffer: %s", vkstrerror(res));
		return;
	}

	res = vkBeginCommandBuffer(cmdbuf, &(VkCommandBufferBeginInfo){
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
			(box[0] / width) / ogbox[0],
			(box[1] / height) / ogbox[1],
			(box[2] / depth) / ogbox[2]);
		mat4x4_translate(A,
			2*off[0] / width,
			2*off[1] / height,
			2*off[2] / depth);
		mat4x4_mul(model, A, model);
		mat4x4_look_at(view,
			(vec3){ 0, 0, -1 },
			(vec3){ 0, 0, 0 },
			(vec3){ 0, -1, 0 });
		mat4x4_perspective(proj, 120, width/height, -1.0, 1.0);
		mat4x4_mul(A, view, model);
		mat4x4_mul(transforms[i], proj, A);
	}
	vkUpdateDescriptorSets(device, 1, (VkWriteDescriptorSet[]){{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = NULL,
			.dstSet = dset,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pImageInfo = NULL,
			.pBufferInfo = (VkDescriptorBufferInfo[]){{
					.buffer = transbuf,
					.offset = 0,
					.range = VK_WHOLE_SIZE,
				},
			},
			.pTexelBufferView = NULL,
		},
	}, 0, NULL);
	vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
		0, 1, &dset, 1, (uint32_t[]){ 0 });
	vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindVertexBuffers(cmdbuf, 0, 1,
		(VkBuffer[]){ vertbuf },
		(VkDeviceSize[]){ 0 });
	vkCmdBeginRenderPass(cmdbuf, &(VkRenderPassBeginInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = renderpass,
		.framebuffer = display.framebuffers[idx],
		.renderArea = {
			.offset = { .x = 0, .y = 0 },
			.extent = { .width = width, .height = height },
		},
		.clearValueCount = 0,
		.pClearValues = NULL,
	}, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdDraw(cmdbuf, meshcnt, 1, 0, 0);
	vkCmdEndRenderPass(cmdbuf);
	vkEndCommandBuffer(cmdbuf);

	VkQueue queue;
	vkGetDeviceQueue(device, qfamidx, 0, &queue);
	res = vkQueueSubmit(queue, 1, &(VkSubmitInfo){
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = (VkSemaphore[]){ acquiresem },
		.pWaitDstStageMask = (VkPipelineStageFlags[]){
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		},
		.commandBufferCount = 1,
		.pCommandBuffers = (VkCommandBuffer[]){
			cmdbuf,
		},
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = (VkSemaphore[]){ display.semaphores[idx] },
	}, fence);
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

	res = vkWaitForFences(
		device, 1, (VkFence[]){ fence }, VK_TRUE, (uint64_t)-1);
	if (res != VK_SUCCESS)
		warnx("couldn't wait on fence");
	vkResetFences(device, 1, (VkFence[]){ fence });

free_dset:
	vkResetDescriptorPool(device, dpool, 0);
}

static int64_t
nanotimerdiff(struct timespec a, struct timespec b)
{
	return 1e9*(a.tv_sec-b.tv_sec) + (a.tv_nsec-b.tv_nsec);
}

int
main(void)
{
	setupcpu();
	setupgpu();
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
	teardown();

	return 0;
}
