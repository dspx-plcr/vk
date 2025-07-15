#include <err.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#define SIZE_T_MAX ((size_t)-1)

static const char *progname = "My Little Vulkan App";

VkInstance instance;
VkDevice device;

const char *
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
	default: return "<bad error code>";
	}
}

char
usabledev(VkPhysicalDeviceProperties2 props)
{
	if (VK_API_VERSION_MAJOR(props.properties.apiVersion) != 1 ||
			VK_API_VERSION_MINOR(props.properties.apiVersion) < 1)
		return 0;
	return 1;
}

char
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

int
main(void)
{
	VkResult res;

	uint32_t apiver;
	res = vkEnumerateInstanceVersion(&apiver);
	if (res != VK_SUCCESS)
		errx(1, "coudln't determine vulkan API version");
	if (VK_API_VERSION_MAJOR(apiver) != 1 ||
			VK_API_VERSION_MINOR(apiver) < 1)
		errx(1, "only compatible with version 1.1 of vulkan");

	res = vkCreateInstance(&(VkInstanceCreateInfo){
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = NULL,
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
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = NULL,
		.enabledExtensionCount = 0,
		.ppEnabledExtensionNames = NULL,
	}, NULL /* allocator */, &instance);
	if (res != VK_SUCCESS)
		errx(1, "couldn't create vulkan instance: %s", vkstrerror(res));

	uint32_t devcnt;
	res = vkEnumeratePhysicalDevices(instance, &devcnt, NULL);
	if (res != VK_SUCCESS)
		errx(1, "coudln't get number of physical devices: %s",
			vkstrerror(res));
	VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * devcnt);
	if (devs == NULL)
		err(1, "couldn't allocate to store physical devices");
	size_t devcap = devcnt;
	while ((res = vkEnumeratePhysicalDevices(instance, &devcnt, devs))
			== VK_INCOMPLETE) {
		if (SIZE_T_MAX / 2 / sizeof(VkPhysicalDevice) > devcap)
			errx(1, "can't store %zu physical devices", devcap);
		devcap *= 2;
		VkPhysicalDevice *tmp =
			realloc(devs, sizeof(VkPhysicalDevice) * devcap);
		if (tmp == NULL)
			err(1, "coudln't allocate to store physical devices");
		devs = tmp;
		devcnt = devcap;
	}

	if (res != VK_SUCCESS)
		errx(1, "couldn't get physical devices: %s", vkstrerror(res));

	VkPhysicalDevice dev;
	VkPhysicalDeviceProperties2 devprops;
	char found = 0;
	for (size_t i = 0; i < devcnt; i++) {
		VkPhysicalDeviceProperties2 props;
		vkGetPhysicalDeviceProperties2(devs[i], &props);
		if (!usabledev(props))
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

	if (!found)
		errx(1, "couldn't find a compatible GPU");

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
	if (res != VK_SUCCESS)
		errx(1, "couldn't create logical device: %s", vkstrerror(res));

	vkDestroyDevice(device, NULL /* allocator */);
	vkDestroyInstance(instance, NULL /* allocator */);
	return 0;
}
