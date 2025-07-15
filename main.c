#include <err.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define ARR_SZ(a) (sizeof(a) / sizeof((a)[0]))
#define SIZE_T_MAX ((size_t)-1)

static const char *progname = "My Little Vulkan App";
static const float width = 1920;
static const float height = 1080;

static VkInstance instance;
static VkDevice device;
static VkRenderPass renderpass;
static VkPipelineLayout layout;
static VkShaderModule vertshader;
static VkShaderModule fragshader;
static VkPipeline pipeline;
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;

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
	DEVICE,
	SURFACE,
	SWAP_CHAIN,
	RENDER_PASS,
	LAYOUT,
	VERT_SHADER,
	FRAG_SHADER,
	SHADERS,
	PIPELINE,
};

static void
cleanup(enum stage s)
{
	switch (s) {
	case PIPELINE:
		vkDestroyPipeline(device, pipeline, NULL /* allocator */);
	case SHADERS:
        case FRAG_SHADER:
		vkDestroyShaderModule(device, fragshader, NULL /* allocator */);
	case VERT_SHADER:
		vkDestroyShaderModule(device, vertshader, NULL /* allocator */);
	case LAYOUT:
		vkDestroyPipelineLayout(device, layout, NULL /* allocator */);
	case RENDER_PASS:
		vkDestroyRenderPass(device, renderpass, NULL /* allocator */);
	case SWAP_CHAIN:
		vkDestroySwapchainKHR(device, swapchain, NULL /* allocator */);
	case SURFACE:
		vkDestroySurfaceKHR(instance, surface, NULL /* allocator */);
	case DEVICE:
		vkDestroyDevice(device, NULL /* allocator */);
	case INSTANCE:
		vkDestroyInstance(instance, NULL /* allocator */);
	}
}

static VkShaderModule
readshader(const char *filename, enum stage toclean)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		cleanup(toclean);
		err(1, "couldn't open shader file for reading");
	}
	if (fseek(f, 0, SEEK_END) < 0) {
		cleanup(toclean);
		err(1, "couldn't seek to end of shader file");
	}
	size_t codesz = ftell(f);
	if (fseek(f, 0, SEEK_SET) < 0) {
		cleanup(toclean);
		err(1, "couldn't seek to start of shader file");
	}
	uint32_t *code = malloc(codesz);
	if (code == NULL) {
		cleanup(toclean);
		err(1, "couldn't allocate for shader code");
	}
	/* TODO: is endianness an issue here? */
	size_t num = fread(code, 1, codesz, f);
	if (num != codesz) {
		cleanup(toclean);
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
		cleanup(toclean);
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

int
main(void)
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
		"VK_KHR_swapchain",
		"VK_KHR_surface",
		"VK_EXT_validation_features",
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

	uint32_t devcnt;
	res = vkEnumeratePhysicalDevices(instance, &devcnt, NULL);
	if (res != VK_SUCCESS) {
		cleanup(INSTANCE);
		errx(1, "coudln't get number of physical devices: %s",
			vkstrerror(res));
	}
	VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * devcnt);
	if (devs == NULL) {
		cleanup(INSTANCE);
		err(1, "couldn't allocate to store physical devices");
	}
	size_t devcap = devcnt;
	while ((res = vkEnumeratePhysicalDevices(instance, &devcnt, devs))
			== VK_INCOMPLETE) {
		if (SIZE_T_MAX / 2 / sizeof(VkPhysicalDevice) > devcap) {
			cleanup(INSTANCE);
			errx(1, "can't store %zu physical devices", devcap);
		}
		devcap *= 2;
		VkPhysicalDevice *tmp =
			realloc(devs, sizeof(VkPhysicalDevice) * devcap);
		if (tmp == NULL) {
			cleanup(INSTANCE);
			err(1, "coudln't allocate to store physical devices");
		}
		devs = tmp;
		devcnt = devcap;
	}

	if (res != VK_SUCCESS) {
		cleanup(INSTANCE);
		errx(1, "couldn't get physical devices: %s", vkstrerror(res));
	}

	VkPhysicalDevice dev;
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
			dev = devs[i];
			devprops = props;
			found = 1;
			continue;
		}

		if (preferdev(props, devprops)) {
			dev = devs[i];
			devprops = props;
		}
	}
	free(devs);

	if (!found) {
		cleanup(INSTANCE);
		errx(1, "couldn't find a compatible GPU");
	}

	res = vkCreateDevice(dev, &(VkDeviceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueCreateInfoCount = 0,
		.pQueueCreateInfos = NULL,
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = NULL,
		.enabledExtensionCount = 0,
		.ppEnabledExtensionNames = NULL,
		.pEnabledFeatures = NULL,
	}, NULL /* allocator */, &device);
	if (res != VK_SUCCESS) {
		cleanup(INSTANCE);
		errx(1, "couldn't create logical device: %s", vkstrerror(res));
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow *window = glfwCreateWindow(
        	width, height, progname, NULL, NULL);
	if (window == NULL) {
        	const char *msg;
        	(void)glfwGetError(&msg);
        	glfwTerminate();
        	cleanup(DEVICE);
        	errx(1, "couldn't create a GLFW window: %s", msg);
	}

	res = glfwCreateWindowSurface(
		instance, window, NULL /* allocator */, &surface);
	if (res != VK_SUCCESS) {
		cleanup(DEVICE);
		errx(1, "couldn't create surface for window: %s",
			vkstrerror(res));
	}

	res = vkCreateSwapchainKHR(device, &(VkSwapchainCreateInfoKHR){
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = NULL,
		.flags = 0,
		.surface = surface,
		.minImageCount = 1,
		.imageFormat = VK_FORMAT_R8G8B8A8_UINT,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		.imageExtent = {
			.width = width,
			.height = height,
		},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = NULL,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_MAILBOX_KHR,
		.clipped = VK_TRUE,
		.oldSwapchain = VK_NULL_HANDLE,
	}, NULL /* allocator */, &swapchain);
	if (res != VK_SUCCESS) {
		cleanup(DEVICE);
		errx(1, "couldn't create swapchain: %s", vkstrerror(res));
	}

	res = vkCreateRenderPass(device, &(VkRenderPassCreateInfo){
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = 0,
		.pAttachments = NULL,
		.subpassCount = 1,
		.pSubpasses = &(VkSubpassDescription){
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = 0,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL,
		},
		.dependencyCount = 0,
		.pDependencies = NULL,
	}, NULL /* allocator */, &renderpass);
	if (res != VK_SUCCESS) {
		cleanup(SWAP_CHAIN);
		errx(1, "coudln't create render pass: %s", vkstrerror(res));
	}

	res = vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = NULL,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL,
	}, NULL /* allocator */, &layout);
	if (res != VK_SUCCESS) {
		cleanup(RENDER_PASS);
		errx(1, "couldn't create pipeline layout: %s", vkstrerror(res));
	}

	vertshader = readshader("vert.spv", LAYOUT);
	fragshader = readshader("frag.spv", VERT_SHADER);

	res = vkCreateGraphicsPipelines(device, NULL /* cache */, 1,
			&(VkGraphicsPipelineCreateInfo){
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		/* TODO: */ .flags = 0,
		.stageCount = 1,
		.pStages = (VkPipelineShaderStageCreateInfo[]){
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			  .pNext = NULL,
			  .flags = 0,
			  .stage = VK_SHADER_STAGE_VERTEX_BIT,
			  .module = vertshader,
			  .pName = "main",
			  .pSpecializationInfo = NULL,
			},
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
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
				.stride = 4*3,
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
			.vertexAttributeDescriptionCount = 1,
			.pVertexAttributeDescriptions = &(VkVertexInputAttributeDescription){
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = 0,
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
				.extent = (VkExtent2D){ .width = 0, .height = 0},
			},
		},
		.pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.depthClampEnable = VK_FALSE,
			/* TODO: */ .rasterizerDiscardEnable = VK_TRUE,
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
			.attachmentCount = 0,
			.pAttachments = NULL,
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
		cleanup(SHADERS);
		errx(1, "couldn't create graphics pipeline: %s",
			vkstrerror(res));
	}

	glfwSetKeyCallback(window, &keycb);
	while (!glfwWindowShouldClose(window))
        	;
	glfwDestroyWindow(window);

	glfwTerminate();
	cleanup(PIPELINE);
	return 0;
}
