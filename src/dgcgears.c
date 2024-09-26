/*
 * Copyright © 2022 Collabora Ltd
 * Copyright © 2024 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

#include <sys/time.h>

#include "vulkan/vulkan.h"

#include "wsi/wsi.h"

#ifndef VK_API_VERSION_MAJOR
/* retain compatibility with old vulkan headers */
#define VK_API_VERSION_MAJOR VK_VERSION_MAJOR
#define VK_API_VERSION_MINOR VK_VERSION_MINOR
#define VK_API_VERSION_PATCH VK_VERSION_PATCH
#endif

static struct wsi_interface wsi;

static VkInstance instance;
static VkPhysicalDevice physical_device;
static VkPhysicalDeviceMemoryProperties mem_props;
static VkDevice device;
static VkQueue queue;

/* swapchain */
static int width, height, new_width, new_height;
static bool fullscreen;
static VkPresentModeKHR desidered_present_mode;
static VkSampleCountFlagBits sample_count;
static uint32_t image_count;
static VkCommandPool cmd_pool;
static VkPresentModeKHR present_mode;
static VkFormat image_format;
static VkColorSpaceKHR color_space;
static VkFormat depth_format;
uint32_t min_image_count = 2;
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;
static VkImage color_msaa, depth_image;
static VkImageView color_msaa_view, depth_view;
static VkDeviceMemory color_msaa_memory, depth_memory;
static VkSemaphore present_semaphore;

struct {
   VkImage image;
   VkImageView view;
} image_data[5];

#define MAX_CONCURRENT_FRAMES 2
struct {
   VkFence fence;
   VkCommandBuffer cmd_buffer;
   VkSemaphore semaphore;
} frame_data[MAX_CONCURRENT_FRAMES];

typedef struct indirect_data {
   uint32_t ies[2];
   VkDrawIndirectCommand draw;
} indirect_data;

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/* gear data */
static VkDescriptorSet descriptor_set;
static VkDeviceMemory ubo_mem;
static VkDeviceMemory vertex_mem;
static VkBuffer ubo_buffer;
static VkBuffer vertex_buffer;
static VkPipelineLayout pipeline_layout;
static VkDescriptorSetLayout set_layout;
static VkPipeline pipeline[3];
size_t vertex_offset, normals_offset;

static VkIndirectCommandsLayoutEXT indirect_layout;
static VkIndirectExecutionSetEXT indirect_execution;
static VkDeviceMemory indirect_mem;
static VkBuffer indirect_buffer;
static VkDeviceAddress indirect_addr;
static VkDeviceMemory preprocess_mem;
static VkBuffer preprocess_buffer;
static VkDeviceSize preprocess_size;
static VkDeviceAddress preprocess_addr;

static PFN_vkCreateIndirectCommandsLayoutEXT CreateIndirectCommandsLayoutEXT;
static PFN_vkCreateIndirectExecutionSetEXT CreateIndirectExecutionSetEXT;
static PFN_vkUpdateIndirectExecutionSetPipelineEXT UpdateIndirectExecutionSetPipelineEXT;
static PFN_vkUpdateIndirectExecutionSetShaderEXT UpdateIndirectExecutionSetShaderEXT;
static PFN_vkGetGeneratedCommandsMemoryRequirementsEXT GetGeneratedCommandsMemoryRequirementsEXT;
static PFN_vkCmdExecuteGeneratedCommandsEXT CmdExecuteGeneratedCommandsEXT;

static VkShaderEXT vs_shaders[3];
static VkShaderEXT fs_shader;
static bool use_shader_object;
static PFN_vkCreateShadersEXT CreateShadersEXT;
static PFN_vkCmdBindShadersEXT CmdBindShadersEXT;
static PFN_vkCmdSetVertexInputEXT CmdSetVertexInputEXT;
static PFN_vkCmdSetPolygonModeEXT CmdSetPolygonModeEXT;
static PFN_vkCmdSetRasterizationSamplesEXT CmdSetRasterizationSamplesEXT;
static PFN_vkCmdSetLogicOpEnableEXT CmdSetLogicOpEnableEXT;
static PFN_vkCmdSetAlphaToCoverageEnableEXT CmdSetAlphaToCoverageEnableEXT;
static PFN_vkCmdSetAlphaToOneEnableEXT CmdSetAlphaToOneEnableEXT;
static PFN_vkCmdSetDepthClampEnableEXT CmdSetDepthClampEnableEXT;
static PFN_vkCmdSetSampleMaskEXT CmdSetSampleMaskEXT;
static PFN_vkCmdSetColorWriteMaskEXT CmdSetColorWriteMaskEXT;
static PFN_vkCmdSetColorBlendEnableEXT CmdSetColorBlendEnableEXT;

static PFN_vkCmdSetPrimitiveTopologyEXT CmdSetPrimitiveTopologyEXT;
static PFN_vkCmdSetPrimitiveRestartEnableEXT CmdSetPrimitiveRestartEnableEXT;
static PFN_vkCmdSetRasterizerDiscardEnableEXT CmdSetRasterizerDiscardEnableEXT;
static PFN_vkCmdSetCullModeEXT CmdSetCullModeEXT;
static PFN_vkCmdSetFrontFaceEXT CmdSetFrontFaceEXT;
static PFN_vkCmdSetDepthTestEnableEXT CmdSetDepthTestEnableEXT;
static PFN_vkCmdSetDepthWriteEnableEXT CmdSetDepthWriteEnableEXT;
static PFN_vkCmdSetDepthCompareOpEXT CmdSetDepthCompareOpEXT;
static PFN_vkCmdSetDepthBoundsTestEnableEXT CmdSetDepthBoundsTestEnableEXT;

struct {
   uint32_t first_vertex;
   uint32_t vertex_count;
} gears[3];

static float view_rot[] = { 20.0, 30.0};
static bool animate = true;

static void
errorv(const char *format, va_list args)
{
   vfprintf(stderr, format, args);
   fprintf(stderr, "\n");
   exit(1);
}

static void
error(const char *format, ...)
{
   va_list args;
   va_start(args, format);
   errorv(format, args);
   va_end(args);
}

static double
current_time(void)
{
   struct timeval tv;
   (void) gettimeofday(&tv, NULL );
   return (double) tv.tv_sec + tv.tv_usec / 1000000.0;
}

static void
init_vk(const char *extension)
{
   VkResult res = vkCreateInstance(
      &(VkInstanceCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pApplicationInfo = &(VkApplicationInfo) {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "dgcgears",
            .apiVersion = VK_MAKE_VERSION(1, 3, 0),
         },
         .enabledExtensionCount = extension ? 2 : 0,
         .ppEnabledExtensionNames = (const char *[2]) {
            VK_KHR_SURFACE_EXTENSION_NAME,
            extension,
         },
      },
      NULL,
      &instance);

   if (res != VK_SUCCESS)
      error("Failed to create Vulkan instance.\n");

   uint32_t count;
   res = vkEnumeratePhysicalDevices(instance, &count, NULL);
   if (res != VK_SUCCESS || count == 0)
      error("No Vulkan devices found.\n");

   VkPhysicalDevice pd[count];
   res = vkEnumeratePhysicalDevices(instance, &count, pd);
   assert(res == VK_SUCCESS);
   physical_device = pd[0];

   vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

   vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
   assert(count > 0);
   VkQueueFamilyProperties props[count];
   vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);
   assert(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT);

   VkPhysicalDeviceShaderObjectFeaturesEXT shobj = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
      .shaderObject = VK_TRUE
   };

   VkPhysicalDeviceVulkan13Features feats13 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      use_shader_object ? &shobj : NULL,
      .dynamicRendering = VK_TRUE
   };

   VkPhysicalDeviceVulkan12Features feats12 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &feats13,
      .bufferDeviceAddress = VK_TRUE
   };
   VkPhysicalDeviceMaintenance5FeaturesKHR maintfeats = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .pNext = &feats12,
      .maintenance5 = VK_TRUE
   };

   VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgcfeats = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
      .pNext = &maintfeats,
      .deviceGeneratedCommands = VK_TRUE
   };
   VkPhysicalDeviceFeatures2 feats2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      &dgcfeats,
      .features = {
         .multiDrawIndirect = VK_TRUE,
      }
   };
   res = vkCreateDevice(physical_device,
      &(VkDeviceCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
         .pNext = &feats2,
         .queueCreateInfoCount = 1,
         .pQueueCreateInfos = &(VkDeviceQueueCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = 0,
            .queueCount = 1,
            .flags = 0,
            .pQueuePriorities = (float []) { 1.0f },
         },
         .enabledExtensionCount = use_shader_object ? 4 : 3,
         .ppEnabledExtensionNames = (const char * const []) {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,
            VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
            VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
         },
      },
      NULL,
      &device);

   if (res != VK_SUCCESS)
      error("Failed to create Vulkan device.\n");

   vkGetDeviceQueue2(device,
      &(VkDeviceQueueInfo2) {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
         .flags = 0,
         .queueFamilyIndex = 0,
         .queueIndex = 0,
      },
      &queue);

   vkCreateCommandPool(device,
      &(const VkCommandPoolCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .queueFamilyIndex = 0,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
      },
      NULL,
      &cmd_pool);

   vkCreateSemaphore(device,
      &(VkSemaphoreCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      },
      NULL,
      &present_semaphore);
   
   CreateIndirectCommandsLayoutEXT = (void*)vkGetDeviceProcAddr(device, "vkCreateIndirectCommandsLayoutEXT");
   CreateIndirectExecutionSetEXT = (void*)vkGetDeviceProcAddr(device, "vkCreateIndirectExecutionSetEXT");
   UpdateIndirectExecutionSetPipelineEXT = (void*)vkGetDeviceProcAddr(device, "vkUpdateIndirectExecutionSetPipelineEXT");
   UpdateIndirectExecutionSetShaderEXT = (void*)vkGetDeviceProcAddr(device, "vkUpdateIndirectExecutionSetShaderEXT");
   GetGeneratedCommandsMemoryRequirementsEXT = (void*)vkGetDeviceProcAddr(device, "vkGetGeneratedCommandsMemoryRequirementsEXT");
   CmdExecuteGeneratedCommandsEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdExecuteGeneratedCommandsEXT");

   CreateShadersEXT  = (void*)vkGetDeviceProcAddr(device, "vkCreateShadersEXT");
   CmdBindShadersEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdBindShadersEXT");
   CmdSetVertexInputEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetVertexInputEXT");
   CmdSetPolygonModeEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetPolygonModeEXT");
   CmdSetRasterizationSamplesEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetRasterizationSamplesEXT");
   CmdSetLogicOpEnableEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetLogicOpEnableEXT");
   CmdSetAlphaToCoverageEnableEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetAlphaToCoverageEnableEXT");
   CmdSetAlphaToOneEnableEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetAlphaToOneEnableEXT");
   CmdSetDepthClampEnableEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetDepthClampEnableEXT");
   CmdSetSampleMaskEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetSampleMaskEXT");
   CmdSetColorWriteMaskEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetColorWriteMaskEXT");
   CmdSetColorBlendEnableEXT  = (void*)vkGetDeviceProcAddr(device, "vkCmdSetColorBlendEnableEXT");


   CmdSetPrimitiveTopologyEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetPrimitiveTopologyEXT");
   CmdSetPrimitiveRestartEnableEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetPrimitiveRestartEnableEXT");
   CmdSetRasterizerDiscardEnableEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetRasterizerDiscardEnableEXT");
   CmdSetCullModeEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetCullModeEXT");
   CmdSetFrontFaceEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetFrontFaceEXT");
   CmdSetDepthTestEnableEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetDepthTestEnableEXT");
   CmdSetDepthWriteEnableEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetDepthWriteEnableEXT");
   CmdSetDepthCompareOpEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetDepthCompareOpEXT");
   CmdSetDepthBoundsTestEnableEXT = (void*)vkGetDeviceProcAddr(device, "vkCmdSetDepthBoundsTestEnableEXT");
}

static int
find_memory_type(const VkMemoryRequirements *reqs,
                 VkMemoryPropertyFlags flags)
{
    for (unsigned i = 0; (1u << i) <= reqs->memoryTypeBits &&
                         i <= mem_props.memoryTypeCount; ++i) {
        if ((reqs->memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return -1;
}

static int
image_allocate(VkImage image, VkMemoryRequirements reqs, int memory_type, VkDeviceMemory *image_memory)
{
   int res = vkAllocateMemory(device,
      &(VkMemoryAllocateInfo) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = reqs.size,
         .memoryTypeIndex = memory_type,
      },
      NULL,
      image_memory);
   if (res != VK_SUCCESS)
      return -1;

   res = vkBindImageMemory(device, image, *image_memory, 0);
   if (res != VK_SUCCESS)
      return -1;

   return 0;
}

static int
create_image(VkFormat format,
             VkExtent3D extent,
             VkSampleCountFlagBits samples,
             VkImageUsageFlags usage,
             VkImage *image)
{
   VkResult res = vkCreateImage(device,
      &(VkImageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .flags = 0,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = format,
         .extent = extent,
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = samples,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = usage,
		   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      }, 0, image);
   if (res != VK_SUCCESS)
      return -1;

   return 0;
}

static int
create_image_view(VkImage image,
                  VkFormat view_format,
                  VkImageAspectFlags aspect_mask,
                  VkImageView *image_view)
{
   int res = vkCreateImageView(device,
      &(VkImageViewCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = view_format,
         .components = {
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
         },
         .subresourceRange = {
            .aspectMask = aspect_mask,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      },
      NULL,
      image_view);

   if(res != VK_SUCCESS)
      return -1;
   return 0;
}

static void
configure_swapchain()
{
   VkSurfaceCapabilitiesKHR surface_caps;
   vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                             &surface_caps);
   assert(surface_caps.supportedCompositeAlpha &
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);

   VkBool32 supported;
   vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, 0, surface,
                                        &supported);
   assert(supported);

   uint32_t count;
   vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                             &count, NULL);
   VkPresentModeKHR present_modes[count];
   vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                             &count, present_modes);
   int i;
   present_mode = VK_PRESENT_MODE_FIFO_KHR;
   for (i = 0; i < count; i++) {
      if (present_modes[i] == desidered_present_mode) {
         present_mode = desidered_present_mode;
         break;
      }
   }

   min_image_count = 2;
   if (min_image_count < surface_caps.minImageCount) {
      if (surface_caps.minImageCount > ARRAY_SIZE(image_data))
          error("surface_caps.minImageCount is too large (is: %d, max: %d)",
                surface_caps.minImageCount, ARRAY_SIZE(image_data));
      min_image_count = surface_caps.minImageCount;
   }

   if (surface_caps.maxImageCount > 0 &&
       min_image_count > surface_caps.maxImageCount) {
      min_image_count = surface_caps.maxImageCount;
   }

   vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface,
                                        &count, NULL);
   VkSurfaceFormatKHR surface_formats[count];
   vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface,
                                        &count, surface_formats);
   image_format = surface_formats[0].format;
   color_space = surface_formats[0].colorSpace;
   for (i = 0; i < count; i++) {
      if (surface_formats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
          surface_formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
         image_format = surface_formats[i].format;
         color_space = surface_formats[i].colorSpace;
         break;
      }
   }

   // either VK_FORMAT_D32_SFLOAT or VK_FORMAT_X8_D24_UNORM_PACK32 needs to be supported; find out which one
   VkFormatProperties props;
   vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_D32_SFLOAT, &props);
   depth_format = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ?
      VK_FORMAT_D32_SFLOAT : VK_FORMAT_X8_D24_UNORM_PACK32;
}

static void
create_swapchain()
{
   vkCreateSwapchainKHR(device,
      &(VkSwapchainCreateInfoKHR) {
         .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
         .flags = 0,
         .surface = surface,
         .minImageCount = min_image_count,
         .imageFormat = image_format,
         .imageColorSpace = color_space,
         .imageExtent = { width, height },
         .imageArrayLayers = 1,
         .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
         .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .queueFamilyIndexCount = 1,
         .pQueueFamilyIndices = (uint32_t[]) { 0 },
         .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
         .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
         .presentMode = present_mode,
      }, NULL, &swapchain);

   vkGetSwapchainImagesKHR(device, swapchain,
                           &image_count, NULL);
   assert(image_count > 0);
   VkImage swapchain_images[image_count];
   vkGetSwapchainImagesKHR(device, swapchain,
                           &image_count, swapchain_images);


   int res;
   if (sample_count != VK_SAMPLE_COUNT_1_BIT) {
       res = create_image(image_format,
         (VkExtent3D) {
            .width = width,
            .height = height,
            .depth = 1,
         },
         sample_count,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
         &color_msaa);
      if (res)
         error("Failed to create resolve image");

      VkMemoryRequirements msaa_reqs;
      vkGetImageMemoryRequirements(device, color_msaa, &msaa_reqs);
      int memory_type = find_memory_type(&msaa_reqs, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
      if (memory_type < 0) {
         memory_type = find_memory_type(&msaa_reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
         if (memory_type < 0)
            error("find_memory_type failed");
      }
      res = image_allocate(color_msaa, msaa_reqs, memory_type, &color_msaa_memory);
      if (res)
         error("Failed to allocate memory for the resolve image");

      res = create_image_view(color_msaa, image_format, VK_IMAGE_ASPECT_COLOR_BIT,
                                        &color_msaa_view);

      if (res)
         error("Failed to create the image view for the resolve image");
   }

   res = create_image(depth_format,
      (VkExtent3D) {
         .width = width,
         .height = height,
         .depth = 1,
      },
      sample_count,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
      &depth_image);

   if (res)
      error("Failed to create depth image");

   VkMemoryRequirements depth_reqs;
   vkGetImageMemoryRequirements(device, depth_image, &depth_reqs);
   int memory_type = find_memory_type(&depth_reqs, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
   if (memory_type < 0) {
      memory_type = find_memory_type(&depth_reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (memory_type < 0)
         error("find_memory_type failed");
   }
   res = image_allocate(depth_image, depth_reqs, memory_type, &depth_memory);
   if (res)
      error("Failed to allocate memory for the depth image");

   res = create_image_view(depth_image,
      depth_format,
      VK_IMAGE_ASPECT_DEPTH_BIT,
      &depth_view);

   if (res)
      error("Failed to create the image view for the depth image");

   for (uint32_t i = 0; i < image_count; i++) {
      image_data[i].image = swapchain_images[i];
      vkCreateImageView(device,
         &(VkImageViewCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = swapchain_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = image_format,
            .components = {
               .r = VK_COMPONENT_SWIZZLE_R,
               .g = VK_COMPONENT_SWIZZLE_G,
               .b = VK_COMPONENT_SWIZZLE_B,
               .a = VK_COMPONENT_SWIZZLE_A,
            },
            .subresourceRange = {
               .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
               .baseMipLevel = 0,
               .levelCount = 1,
               .baseArrayLayer = 0,
               .layerCount = 1,
            },
         },
         NULL,
         &image_data[i].view);
   }

   for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
      vkCreateFence(device,
         &(VkFenceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
         },
         NULL,
         &frame_data[i].fence);

      vkAllocateCommandBuffers(device,
         &(VkCommandBufferAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
         },
         &frame_data[i].cmd_buffer);

      vkCreateSemaphore(device,
         &(VkSemaphoreCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
         },
         NULL,
         &frame_data[i].semaphore);
   }
}

static void
free_swapchain_data()
{
   for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      vkFreeCommandBuffers(device, cmd_pool, 1, &frame_data[i].cmd_buffer);
      vkDestroyFence(device, frame_data[i].fence, NULL);
      vkDestroySemaphore(device, frame_data[i].semaphore, NULL);
   }

   for (uint32_t i = 0; i < image_count; i++) {
      vkDestroyImageView(device, image_data[i].view, NULL);
   }

   vkDestroyImageView(device, depth_view, NULL);
   vkDestroyImage(device, depth_image, NULL);
   vkFreeMemory(device, depth_memory, NULL);

   if (sample_count != VK_SAMPLE_COUNT_1_BIT) {
      vkDestroyImageView(device, color_msaa_view, NULL);
      vkDestroyImage(device, color_msaa, NULL);
      vkFreeMemory(device, color_msaa_memory, NULL);
   }
}

static void
recreate_swapchain()
{
   vkDeviceWaitIdle(device);
   free_swapchain_data();
   vkDestroySwapchainKHR(device, swapchain, NULL);
   width = new_width, height = new_height;
   create_swapchain();
}

static VkBuffer
create_buffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
   VkBuffer buffer;

   VkResult result =
      vkCreateBuffer(device,
         &(VkBufferCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .flags = 0
         },
         NULL,
         &buffer);

   if (result != VK_SUCCESS)
      error("Failed to create buffer");

   return buffer;
}

static VkDeviceMemory
allocate_buffer_mem(VkBuffer buffer, VkDeviceSize mem_size)
{
   VkMemoryRequirements reqs;
   vkGetBufferMemoryRequirements(device, buffer, &reqs);

   int memory_type = find_memory_type(&reqs,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (memory_type < 0)
      error("failed to find coherent memory type");

   VkDeviceMemory mem;
   VkMemoryAllocateFlagsInfo info = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   vkAllocateMemory(device,
      &(VkMemoryAllocateInfo) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .pNext = &info,
         .allocationSize = mem_size,
         .memoryTypeIndex = memory_type,
      },
      NULL,
      &mem);
   return mem;
}

static uint32_t red_spirv_source[] = {
#include "red.vert.spv.h"
};

static uint32_t green_spirv_source[] = {
#include "green.vert.spv.h"
};

static uint32_t blue_spirv_source[] = {
#include "blue.vert.spv.h"
};

static uint32_t fs_spirv_source[] = {
#include "gear.frag.spv.h"
};

struct ubo {
   float projection[16];
};

struct push_constants {
   float angle;
   float view_rot_0;
   float view_rot_1;
   float h;
};

struct gear {
   int nvertices;
};

#define GEAR_VERTEX_STRIDE 6

static int
create_gear(float verts[],
            float inner_radius, float outer_radius, float width,
            int teeth, float tooth_depth)
{
   // normal state
   float current_normal[3];
#define SET_NORMAL(x, y, z) do { \
      current_normal[0] = x; \
      current_normal[1] = y; \
      current_normal[2] = z; \
   } while (0)

   // vertex buffer state
   unsigned num_verts = 0;

#define EMIT_VERTEX(x, y, z) do { \
      verts[num_verts * GEAR_VERTEX_STRIDE + 0] = x; \
      verts[num_verts * GEAR_VERTEX_STRIDE + 1] = y; \
      verts[num_verts * GEAR_VERTEX_STRIDE + 2] = z; \
      memcpy(verts + num_verts * GEAR_VERTEX_STRIDE + 3, \
             current_normal, sizeof(current_normal)); \
      num_verts++; \
   } while (0)

   // strip restart-logic
   int cur_strip_start = 0;
#define START_STRIP() do { \
   cur_strip_start = num_verts; \
   if (cur_strip_start) \
      num_verts += 2; \
} while(0);

#define END_STRIP() do { \
   if (cur_strip_start) { \
      memcpy(verts + cur_strip_start * GEAR_VERTEX_STRIDE, \
             verts + (cur_strip_start - 1) * GEAR_VERTEX_STRIDE, \
             sizeof(float) * GEAR_VERTEX_STRIDE); \
      memcpy(verts + (cur_strip_start + 1) * GEAR_VERTEX_STRIDE, \
             verts + (cur_strip_start + 2) * GEAR_VERTEX_STRIDE, \
             sizeof(float) * GEAR_VERTEX_STRIDE); \
   } \
} while (0)

   float r0 = inner_radius;
   float r1 = outer_radius - tooth_depth / 2.0;
   float r2 = outer_radius + tooth_depth / 2.0;

   float da = 2.0 * M_PI / teeth / 4.0;

   SET_NORMAL(0.0, 0.0, 1.0);

   /* draw front face */
   START_STRIP();
   for (int i = 0; i <= teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), width * 0.5);
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), width * 0.5);
      if (i < teeth) {
         EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), width * 0.5);
         EMIT_VERTEX(r1 * cos(angle + 3 * da),
                     r1 * sin(angle + 3 * da), width * 0.5);
      }
   }
   END_STRIP();

   /* draw front sides of teeth */
   for (int i = 0; i < teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      START_STRIP();
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da), width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
		 width * 0.5);
      END_STRIP();
   }

   SET_NORMAL(0.0, 0.0, -1.0);

   /* draw back face */
   START_STRIP();
   for (int i = 0; i <= teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), -width * 0.5);
      EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), -width * 0.5);
      if (i < teeth) {
         EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
               -width * 0.5);
         EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), -width * 0.5);
      }
   }
   END_STRIP();

   /* draw back sides of teeth */
   for (int i = 0; i < teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      START_STRIP();
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 -width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
		 -width * 0.5);
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), -width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da), -width * 0.5);
      END_STRIP();
   }

   /* draw outward faces of teeth */
   for (int i = 0; i < teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      float u = r2 * cos(angle + da) - r1 * cos(angle);
      float v = r2 * sin(angle + da) - r1 * sin(angle);
      float len = sqrt(u * u + v * v);
      u /= len;
      v /= len;
      SET_NORMAL(v, -u, 0.0);
      START_STRIP();
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), width * 0.5);
      EMIT_VERTEX(r1 * cos(angle), r1 * sin(angle), -width * 0.5);

      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da), width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da), -width * 0.5);
      END_STRIP();

      SET_NORMAL(cos(angle), sin(angle), 0.0);
      START_STRIP();
      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da),
		 width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + da), r2 * sin(angle + da),
		 -width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
      width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
      -width * 0.5);
      END_STRIP();

      u = r1 * cos(angle + 3 * da) - r2 * cos(angle + 2 * da);
      v = r1 * sin(angle + 3 * da) - r2 * sin(angle + 2 * da);
      SET_NORMAL(v, -u, 0.0);
      START_STRIP();
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
		 width * 0.5);
      EMIT_VERTEX(r2 * cos(angle + 2 * da), r2 * sin(angle + 2 * da),
		 -width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 -width * 0.5);
      END_STRIP();

      SET_NORMAL(cos(angle), sin(angle), 0.0);
      START_STRIP();
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 3 * da), r1 * sin(angle + 3 * da),
		 -width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 4 * da), r1 * sin(angle + 4 * da),
      width * 0.5);
      EMIT_VERTEX(r1 * cos(angle + 4 * da), r1 * sin(angle + 4 * da),
      -width * 0.5);
      END_STRIP();
   }

   /* draw inside radius cylinder */
   START_STRIP();
   for (int i = 0; i <= teeth; i++) {
      float angle = i * 2.0 * M_PI / teeth;
      SET_NORMAL(-cos(angle), -sin(angle), 0.0);
      EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), -width * 0.5);
      EMIT_VERTEX(r0 * cos(angle), r0 * sin(angle), width * 0.5);
   }
   END_STRIP();

   return num_verts;
}


static void
init_gears()
{
   VkResult r;

   vkCreateDescriptorSetLayout(device,
      &(VkDescriptorSetLayoutCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1,
         .pBindings = (VkDescriptorSetLayoutBinding[]) {
            {
               .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               .descriptorCount = 1,
               .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
               .pImmutableSamplers = NULL
            }
         }
      },
      NULL,
      &set_layout);

   vkCreatePipelineLayout(device,
      &(VkPipelineLayoutCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &set_layout,
         .pPushConstantRanges = (VkPushConstantRange[]) {
            {
               .offset = 0,
               .size = sizeof(struct push_constants),
               .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            },
         },
         .pushConstantRangeCount = 1,
      },
      NULL,
      &pipeline_layout);

   if (use_shader_object) {
      VkShaderEXT shaders[4];
      CreateShadersEXT(device, 4,
         (VkShaderCreateInfoEXT[]) {
            {
               VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
               NULL,
               VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT,
               VK_SHADER_STAGE_VERTEX_BIT,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               VK_SHADER_CODE_TYPE_SPIRV_EXT,
               sizeof(red_spirv_source),
               red_spirv_source,
               "main",
               1,
               &set_layout,
               1,
               (VkPushConstantRange[]) {
               {
                  .offset = 0,
                  .size = sizeof(struct push_constants),
                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
               }},
               NULL
            },
            {
               VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
               NULL,
               VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT,
               VK_SHADER_STAGE_VERTEX_BIT,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               VK_SHADER_CODE_TYPE_SPIRV_EXT,
               sizeof(green_spirv_source),
               green_spirv_source,
               "main",
               1,
               &set_layout,
               1,
               (VkPushConstantRange[]) {
               {
                  .offset = 0,
                  .size = sizeof(struct push_constants),
                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
               }},
               NULL
            },
            {
               VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
               NULL,
               VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT,
               VK_SHADER_STAGE_VERTEX_BIT,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               VK_SHADER_CODE_TYPE_SPIRV_EXT,
               sizeof(blue_spirv_source),
               blue_spirv_source,
               "main",
               1,
               &set_layout,
               1,
               (VkPushConstantRange[]) {
               {
                  .offset = 0,
                  .size = sizeof(struct push_constants),
                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
               }},
               NULL
            },
            {
               VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
               NULL,
               VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT,
               VK_SHADER_STAGE_FRAGMENT_BIT,
               0,
               VK_SHADER_CODE_TYPE_SPIRV_EXT,
               sizeof(fs_spirv_source),
               fs_spirv_source,
               "main",
               1,
               &set_layout,
               1,
               (VkPushConstantRange[]) {
               {
                  .offset = 0,
                  .size = sizeof(struct push_constants),
                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
               }},
               NULL
            },
         },
         NULL,
         shaders
      );
      for (int i = 0; i < 3; i++)
         vs_shaders[i] = shaders[i];
      fs_shader = shaders[3];
   } else {
      VkShaderModule red_module;
      vkCreateShaderModule(device,
         &(VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(red_spirv_source),
            .pCode = red_spirv_source,
         },
         NULL,
         &red_module);
      VkShaderModule green_module;
      vkCreateShaderModule(device,
         &(VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(green_spirv_source),
            .pCode = green_spirv_source,
         },
         NULL,
         &green_module);
      VkShaderModule blue_module;
      vkCreateShaderModule(device,
         &(VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(blue_spirv_source),
            .pCode = blue_spirv_source,
         },
         NULL,
         &blue_module);
      VkShaderModule vs_modules[] = {red_module, green_module, blue_module};

      VkShaderModule fs_module;
      vkCreateShaderModule(device,
         &(VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(fs_spirv_source),
            .pCode = fs_spirv_source,
         },
         NULL,
         &fs_module);

      VkPipelineRenderingCreateInfo pci = {
         VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
         NULL,
         0,
         1,
         &image_format,
         depth_format,
      };
      VkPipelineCreateFlags2CreateInfoKHR pipeline2 = {
         VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
         &pci,
         VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT
      };
      for (unsigned i = 0; i < ARRAY_SIZE(vs_modules); i++) {
         vkCreateGraphicsPipelines(device,
            (VkPipelineCache) { VK_NULL_HANDLE },
            1,
            &(VkGraphicsPipelineCreateInfo) {
               .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
               .pNext = &pipeline2,
               .stageCount = 2,
               .pStages = (VkPipelineShaderStageCreateInfo[]) {
                     {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage = VK_SHADER_STAGE_VERTEX_BIT,
                        .module = vs_modules[i],
                        .pName = "main",
                     },
                     {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                        .module = fs_module,
                        .pName = "main",
                     },
               },
               .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                  .vertexBindingDescriptionCount = 2,
                  .pVertexBindingDescriptions = (VkVertexInputBindingDescription[]) {
                     {
                        .binding = 0,
                        .stride = 6 * sizeof(float),
                        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
                     },
                     {
                        .binding = 1,
                        .stride = 6 * sizeof(float),
                        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
                     },
                  },
                  .vertexAttributeDescriptionCount = 2,
                  .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]) {
                     {
                        .location = 0,
                        .binding = 0,
                        .format = VK_FORMAT_R32G32B32_SFLOAT,
                        .offset = 0
                     },
                     {
                        .location = 1,
                        .binding = 1,
                        .format = VK_FORMAT_R32G32B32_SFLOAT,
                        .offset = 0
                     },
                  }
               },
               .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                  .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                  .primitiveRestartEnable = false,
               },

               .pViewportState = &(VkPipelineViewportStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                  .viewportCount = 1,
                  .scissorCount = 1,
               },

               .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                  .rasterizerDiscardEnable = false,
                  .polygonMode = VK_POLYGON_MODE_FILL,
                  .cullMode = VK_CULL_MODE_BACK_BIT,
                  .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                  .lineWidth = 1.0f,
               },

               .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                  .rasterizationSamples = sample_count,
               },
               .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                  .depthTestEnable = VK_TRUE,
                  .depthWriteEnable = VK_TRUE,
                  .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
               },

               .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                  .attachmentCount = 1,
                  .pAttachments = (VkPipelineColorBlendAttachmentState []) {
                     { .colorWriteMask = VK_COLOR_COMPONENT_A_BIT |
                                          VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT },
                  }
               },

               .pDynamicState = &(VkPipelineDynamicStateCreateInfo) {
                  .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                  .dynamicStateCount = 2,
                  .pDynamicStates = (VkDynamicState[]) {
                        VK_DYNAMIC_STATE_VIEWPORT,
                        VK_DYNAMIC_STATE_SCISSOR,
                  },
               },

               .flags = 0,
               .layout = pipeline_layout,
               .subpass = 0,
               .basePipelineHandle = (VkPipeline) { 0 },
               .basePipelineIndex = 0
            },
            NULL,
            &pipeline[i]);
      }
   }

   CreateIndirectCommandsLayoutEXT(device,
                                     &(VkIndirectCommandsLayoutCreateInfoEXT) {
                                       .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
                                       .flags = 0,
                                       .shaderStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       .indirectStride = sizeof(indirect_data),
                                       .pipelineLayout = pipeline_layout,
                                       .tokenCount = 2,
                                       .pTokens = (VkIndirectCommandsLayoutTokenEXT[]) {
                                          {
                                             .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
                                             .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT,
                                             .data = {
                                                .pExecutionSet = &(VkIndirectCommandsExecutionSetTokenEXT) {
                                                   .type = use_shader_object ? VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT : VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT,
                                                   .shaderStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                }
                                             },
                                             .offset = 0
                                          },
                                          {
                                             .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
                                             .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_EXT,
                                             .offset = offsetof(indirect_data, draw)
                                          },
                                       }
                                     },
                                     NULL, &indirect_layout);

   if (use_shader_object) {
      CreateIndirectExecutionSetEXT(device,
                                       &(VkIndirectExecutionSetCreateInfoEXT) {
                                          .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT,
                                          .type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT,
                                          .info = {
                                             .pShaderInfo = &(VkIndirectExecutionSetShaderInfoEXT) {
                                                .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_INFO_EXT,
                                                .shaderCount = 2,
                                                .pInitialShaders = (VkShaderEXT[]) {
                                                   vs_shaders[0],
                                                   fs_shader,
                                                },
                                                .pSetLayoutInfos = (VkIndirectExecutionSetShaderLayoutInfoEXT[]) {
                                                   {
                                                      .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT,
                                                      .setLayoutCount = 1,
                                                      .pSetLayouts = (VkDescriptorSetLayout[]) {
                                                         set_layout
                                                      }
                                                   },
                                                   {
                                                      .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT,
                                                      .setLayoutCount = 0,
                                                      .pSetLayouts = (VkDescriptorSetLayout[]) {
                                                      }
                                                   },
                                                },
                                                .maxShaderCount = 4,
                                                .pushConstantRangeCount = 1,
                                                .pPushConstantRanges = (VkPushConstantRange[]) {
                                                   {
                                                      .offset = 0,
                                                      .size = sizeof(struct push_constants),
                                                      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                                                   },
                                                },
                                             },
                                          },
                                       },
                                       NULL, &indirect_execution);
      UpdateIndirectExecutionSetShaderEXT(device, indirect_execution,
                                                2, (VkWriteIndirectExecutionSetShaderEXT[]) {
                                                   {
                                                   .sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT,
                                                   .index = 2,
                                                   .shader = vs_shaders[1],
                                                   },
                                                   {
                                                   .sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT,
                                                   .index = 3,
                                                   .shader = vs_shaders[2],
                                                   },
                                                });
   } else {
      CreateIndirectExecutionSetEXT(device,
                                       &(VkIndirectExecutionSetCreateInfoEXT) {
                                          .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT,
                                          .type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT,
                                          .info = {
                                             .pPipelineInfo = &(VkIndirectExecutionSetPipelineInfoEXT) {
                                                .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT,
                                                .initialPipeline = pipeline[0],
                                                .maxPipelineCount = 3
                                             },
                                          },
                                       },
                                       NULL, &indirect_execution);
      UpdateIndirectExecutionSetPipelineEXT(device, indirect_execution,
                                                2, (VkWriteIndirectExecutionSetPipelineEXT[]) {
                                                   {
                                                   .sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT,
                                                   .index = 1,
                                                   .pipeline = pipeline[1],
                                                   },
                                                   {
                                                   .sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT,
                                                   .index = 2,
                                                   .pipeline = pipeline[2],
                                                   },
                                                });

   }

   VkMemoryRequirements2 memreqs = {
      VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   GetGeneratedCommandsMemoryRequirementsEXT(device,
                                               &(VkGeneratedCommandsMemoryRequirementsInfoEXT) {
                                                  .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
                                                  .indirectExecutionSet = indirect_execution,
                                                  .indirectCommandsLayout = indirect_layout,
                                                  .maxSequenceCount = 3,
                                               },
                                               &memreqs);

   VkBufferUsageFlags2CreateInfoKHR busage = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
      .usage = VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR,
   };
   vkCreateBuffer(device,
      &(VkBufferCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .pNext = &busage,
         .size = memreqs.memoryRequirements.size,
      },
      NULL,
      &preprocess_buffer);
   VkMemoryAllocateFlagsInfo info = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   vkAllocateMemory(device,
      &(VkMemoryAllocateInfo) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .pNext = &info,
         .allocationSize = memreqs.memoryRequirements.size,
         .memoryTypeIndex = ffs(memreqs.memoryRequirements.memoryTypeBits) - 1,
      },
      NULL,
      &preprocess_mem);
   vkBindBufferMemory(device, preprocess_buffer, preprocess_mem, 0);
   preprocess_size = memreqs.memoryRequirements.size;
   preprocess_addr = vkGetBufferDeviceAddress(device,
                                              &(VkBufferDeviceAddressInfo) {
                                                 .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                 .buffer = preprocess_buffer
                                              });

#define MAX_VERTS 10000
   float verts[MAX_VERTS * GEAR_VERTEX_STRIDE];

   gears[0].first_vertex = 0;
   gears[0].vertex_count = create_gear(verts + gears[0].first_vertex * GEAR_VERTEX_STRIDE,
                                       1.0, 4.0, 1.0, 20, 0.7);
   gears[1].first_vertex = gears[0].first_vertex + gears[0].vertex_count;
   gears[1].vertex_count = create_gear(verts + gears[1].first_vertex * GEAR_VERTEX_STRIDE,
                                       0.5, 2.0, 2.0, 10, 0.7);
   gears[2].first_vertex = gears[1].first_vertex + gears[1].vertex_count;
   gears[2].vertex_count = create_gear(verts + gears[2].first_vertex * GEAR_VERTEX_STRIDE,
                                       1.3, 2.0, 0.5, 10, 0.7);

   unsigned num_verts = gears[2].first_vertex + gears[2].vertex_count;
   unsigned mem_size = sizeof(float) * GEAR_VERTEX_STRIDE * num_verts;
   vertex_offset = 0;
   normals_offset = sizeof(float) * 3;
   ubo_buffer = create_buffer(sizeof(struct ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   vertex_buffer = create_buffer(mem_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

   ubo_mem = allocate_buffer_mem(ubo_buffer, sizeof(struct ubo));
   vertex_mem = allocate_buffer_mem(vertex_buffer, mem_size);

   size_t indirect_size = 3 * (sizeof(uint32_t) + sizeof(VkDrawIndirectCommand));
   indirect_buffer = create_buffer(indirect_size, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
   indirect_mem = allocate_buffer_mem(indirect_buffer, indirect_size);

   indirect_data *indirect_map;
   // VkDrawIndirectCommand *indirect;
   // vkMapMemory(device, indirect_mem, 0, indirect_size, 0, (void*)&indirect);
   // for (unsigned i = 0; i < ARRAY_SIZE(gears); i++) {
   //    indirect[i].vertexCount = gears[i].vertex_count;
   //    indirect[i].firstVertex = gears[i].first_vertex;
   //    indirect[i].firstInstance = 0;
   //    indirect[i].instanceCount = 1;
   // }
   vkMapMemory(device, indirect_mem, 0, indirect_size, 0, (void*)&indirect_map);
   int pipeline_idx[] = {
      0, 1, 2
   };
   /* VS are offset */
   int shader_idx[] = {
      0, 2, 3
   };
   for (unsigned i = 0; i < ARRAY_SIZE(gears); i++) {
      indirect_map[i].ies[0] = use_shader_object ? shader_idx[i] : pipeline_idx[i];
      indirect_map[i].ies[1] = 1;
      indirect_map[i].draw.vertexCount = gears[i].vertex_count;
      indirect_map[i].draw.firstVertex = gears[i].first_vertex;
      indirect_map[i].draw.firstInstance = 0;
      indirect_map[i].draw.instanceCount = 1;
   }
   vkUnmapMemory(device, indirect_mem);
   vkBindBufferMemory(device, indirect_buffer, indirect_mem, 0);
   indirect_addr = vkGetBufferDeviceAddress(device,
                                             &(VkBufferDeviceAddressInfo) {
                                                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                .buffer = indirect_buffer
                                             });

   void *map;
   r = vkMapMemory(device, vertex_mem, 0, mem_size, 0, &map);
   if (r != VK_SUCCESS)
      error("vkMapMemory failed");
   memcpy(map, verts, mem_size);
   vkUnmapMemory(device, vertex_mem);

   vkBindBufferMemory(device, ubo_buffer, ubo_mem, 0);
   vkBindBufferMemory(device, vertex_buffer, vertex_mem, 0);

   VkDescriptorPool desc_pool;
   const VkDescriptorPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = (VkDescriptorPoolSize[]) {
         {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1
         },
      }
   };

   vkCreateDescriptorPool(device, &create_info, NULL, &desc_pool);

   vkAllocateDescriptorSets(device,
      &(VkDescriptorSetAllocateInfo) {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = desc_pool,
         .descriptorSetCount = 1,
         .pSetLayouts = &set_layout,
      }, &descriptor_set);

   vkUpdateDescriptorSets(device, 1,
      (VkWriteDescriptorSet []) {
         {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo) {
               .buffer = ubo_buffer,
               .offset = 0,
               .range = sizeof(struct ubo),
            }
         }
      },
      0, NULL);
}

float angle = 0.0;

#define G2L(x) ((x) < 0.04045 ? (x) / 12.92 : powf(((x) + 0.055) / 1.055, 2.4))

static void
draw_gears(VkCommandBuffer cmdbuf)
{
   vkCmdBindVertexBuffers(cmdbuf, 0, 2,
      (VkBuffer[]) {
         vertex_buffer,
         vertex_buffer,
      },
      (VkDeviceSize[]) {
         vertex_offset,
         normals_offset
      });

   if (use_shader_object)
      CmdBindShadersEXT(cmdbuf, 2, (VkShaderStageFlagBits[]) { VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT }, (VkShaderEXT[]){vs_shaders[0], fs_shader});
   else
      vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline[0]);

   vkCmdBindDescriptorSets(cmdbuf,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipeline_layout,
      0, 1,
      &descriptor_set, 0, NULL);


   
   if (use_shader_object) {
      vkCmdSetViewportWithCount(cmdbuf, 1,
         (VkViewport[]) {
            {
               .x = 0,
               .y = 0,
               .width = width,
               .height = height,
               .minDepth = 0,
               .maxDepth = 1,
            }
         });

      vkCmdSetScissorWithCount(cmdbuf, 1,
         (VkRect2D[]) {
            {
               .offset = { 0, 0 },
               .extent = { width, height },
            }
         });
      CmdSetVertexInputEXT(cmdbuf,
            2, (VkVertexInputBindingDescription2EXT[]) {
            {
               .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
               .binding = 0,
               .stride = 6 * sizeof(float),
               .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
               .divisor = 1,
            },
            {
               .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
               .binding = 1,
               .stride = 6 * sizeof(float),
               .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
               .divisor = 1,
            },
         },
         2, (VkVertexInputAttributeDescription2EXT[]) {
            {
               .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
               .location = 0,
               .binding = 0,
               .format = VK_FORMAT_R32G32B32_SFLOAT,
               .offset = 0
            },
            {
               .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
               .location = 1,
               .binding = 1,
               .format = VK_FORMAT_R32G32B32_SFLOAT,
               .offset = 0
            },
         }
      );
      CmdSetPrimitiveTopologyEXT(cmdbuf, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
      CmdSetPrimitiveRestartEnableEXT(cmdbuf, VK_FALSE);
      CmdSetRasterizerDiscardEnableEXT(cmdbuf, VK_FALSE);
      CmdSetCullModeEXT(cmdbuf, VK_CULL_MODE_BACK_BIT);
      CmdSetFrontFaceEXT(cmdbuf, VK_FRONT_FACE_COUNTER_CLOCKWISE);
      CmdSetDepthTestEnableEXT(cmdbuf, VK_TRUE);
      CmdSetDepthWriteEnableEXT(cmdbuf, VK_TRUE);
      CmdSetDepthCompareOpEXT(cmdbuf, VK_COMPARE_OP_LESS_OR_EQUAL);
      CmdSetDepthBoundsTestEnableEXT(cmdbuf, VK_FALSE);
      CmdSetPolygonModeEXT(cmdbuf, VK_POLYGON_MODE_FILL);
      CmdSetRasterizationSamplesEXT(cmdbuf, sample_count);
      CmdSetLogicOpEnableEXT(cmdbuf, VK_FALSE);
      CmdSetAlphaToCoverageEnableEXT(cmdbuf, VK_FALSE);
      CmdSetAlphaToOneEnableEXT(cmdbuf, VK_FALSE);
      CmdSetDepthClampEnableEXT(cmdbuf, VK_FALSE);
      CmdSetSampleMaskEXT(cmdbuf, sample_count, (VkSampleMask[]){UINT32_MAX});
      CmdSetColorWriteMaskEXT(cmdbuf, 0, 1, (VkColorComponentFlags[]){ VK_COLOR_COMPONENT_A_BIT |
                                          VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT});
      CmdSetColorBlendEnableEXT(cmdbuf, 0, 1, (VkBool32[]){VK_FALSE});
   } else {
      vkCmdSetViewport(cmdbuf, 0, 1,
         &(VkViewport) {
            .x = 0,
            .y = 0,
            .width = width,
            .height = height,
            .minDepth = 0,
            .maxDepth = 1,
         });

      vkCmdSetScissor(cmdbuf, 0, 1,
         &(VkRect2D) {
            .offset = { 0, 0 },
            .extent = { width, height },
         });
   }

   struct push_constants push_constants;
   push_constants.angle = angle;
   push_constants.view_rot_0 = view_rot[0];
   push_constants.view_rot_1 = view_rot[1];
   push_constants.h = (float)height / width;

   vkCmdPushConstants(cmdbuf, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                      0, sizeof(push_constants), &push_constants);

   // vkCmdDraw(cmdbuf, gears[0].vertex_count, 1, gears[0].first_vertex, 0);
   // vkCmdDrawIndirect(cmdbuf, indirect_buffer, 0, 1, 0);
   CmdExecuteGeneratedCommandsEXT(cmdbuf, VK_FALSE,
                                    &(VkGeneratedCommandsInfoEXT) {
                                       .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
                                       .shaderStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       .indirectExecutionSet = indirect_execution,
                                       .indirectCommandsLayout = indirect_layout,
                                       .indirectAddress = indirect_addr,
                                       .indirectAddressSize = 3 * sizeof(indirect_data),
                                       .preprocessAddress = preprocess_addr,
                                       .preprocessSize = preprocess_size,
                                       .maxSequenceCount = 3,
                                    });
}

static const char *
get_devtype_str(VkPhysicalDeviceType devtype)
{
   static char buf[256];
   switch (devtype) {
   case VK_PHYSICAL_DEVICE_TYPE_OTHER:
      return "other";
   case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated GPU";
   case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete GPU";
   case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual GPU";
   case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
   default:
      snprintf(buf, sizeof(buf), "Unknown (%08x)", devtype);
      return buf;
   }
}

static void
usage(void)
{
   printf("Usage:\n");
   printf("  -samples N              run in multisample mode with N samples\n");
   printf("  -present-mailbox        run with present mode mailbox\n");
   printf("  -present-immediate      run with present mode immediate\n");
   printf("  -fullscreen             run in fullscreen mode\n");
   printf("  -info                   display Vulkan device info\n");
   printf("  -size WxH               window size\n");
}

static void
print_info()
{
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical_device, &properties);
   printf("apiVersion       = %d.%d.%d\n",
            VK_API_VERSION_MAJOR(properties.apiVersion),
            VK_API_VERSION_MINOR(properties.apiVersion),
            VK_API_VERSION_PATCH(properties.apiVersion));
   printf("driverVersion    = %04x\n", properties.driverVersion);
   printf("vendorID         = %04x\n", properties.vendorID);
   printf("deviceID         = %04x\n", properties.deviceID);
   printf("deviceType       = %s\n", get_devtype_str(properties.deviceType));
   printf("deviceName       = %s\n", properties.deviceName);

   uint32_t num_extensions = 0;
   VkExtensionProperties *extensions;
   vkEnumerateDeviceExtensionProperties(physical_device, NULL, &num_extensions, NULL);
   if (num_extensions > 0) {
      extensions = calloc(num_extensions, sizeof(VkExtensionProperties));
      if (!extensions)
         error("Failed to allocate memory");

      vkEnumerateDeviceExtensionProperties(physical_device, NULL, &num_extensions, extensions);
      printf("deviceExtensions =\n");
      for (int i = 0; i < num_extensions; ++i)
         printf("\t%s\n", extensions[i].extensionName);
   }
}

static VkSampleCountFlagBits
sample_count_flag(int sample_count)
{
   switch (sample_count) {
   case 1:
      return VK_SAMPLE_COUNT_1_BIT;
   case 2:
      return VK_SAMPLE_COUNT_2_BIT;
   case 4:
      return VK_SAMPLE_COUNT_4_BIT;
   case 8:
      return VK_SAMPLE_COUNT_8_BIT;
   case 16:
      return VK_SAMPLE_COUNT_16_BIT;
   case 32:
      return VK_SAMPLE_COUNT_32_BIT;
   case 64:
      return VK_SAMPLE_COUNT_64_BIT;
   default:
      error("Invalid sample count");
   }
   return VK_SAMPLE_COUNT_1_BIT;
}

static bool
check_sample_count_support(VkSampleCountFlagBits sample_count)
{
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical_device, &properties);

   return (sample_count & properties.limits.framebufferColorSampleCounts) &&
      (sample_count & properties.limits.framebufferDepthSampleCounts);
}

static void
wsi_resize(int p_new_width, int p_new_height)
{
   new_width = p_new_width;
   new_height = p_new_height;
}

static void
wsi_key_press(bool down, enum wsi_key key) {
   if (!down)
      return;
   switch (key) {
      case WSI_KEY_ESC:
         exit(0);
      case WSI_KEY_UP:
         view_rot[0] += 5.0;
         break;
      case WSI_KEY_DOWN:
         view_rot[0] -= 5.0;
         break;
      case WSI_KEY_LEFT:
         view_rot[1] += 5.0;
         break;
      case WSI_KEY_RIGHT:
         view_rot[1] -= 5.0;
         break;
      case WSI_KEY_A:
         animate = !animate;
         break;
      default:
         break;
   }
}

static void
wsi_exit()
{
   exit(0);
}

static struct wsi_callbacks wsi_callbacks = {
   .resize = wsi_resize,
   .key_press = wsi_key_press,
   .exit = wsi_exit,
};

static void
buffer_barrier(VkCommandBuffer cmd_buffer,
               VkPipelineStageFlags src_flags,
               VkPipelineStageFlags dst_flags,
               VkAccessFlags src_access,
               VkAccessFlags dst_access,
               VkBuffer buffer,
               VkDeviceSize offset,
               VkDeviceSize size)
{
   vkCmdPipelineBarrier(cmd_buffer,
      src_flags, dst_flags,
      0, 0, NULL,
      1, &(VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .pNext = NULL,
         .srcAccessMask = src_access,
         .dstAccessMask = dst_access,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = buffer,
         .offset = offset,
         .size = size,
      },
      0, NULL);
}

int
main(int argc, char *argv[])
{
   bool printInfo = false;
   sample_count = VK_SAMPLE_COUNT_1_BIT;
   desidered_present_mode = VK_PRESENT_MODE_FIFO_KHR;
   width = 300;
   height = 300;
   fullscreen = false;

   for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-info") == 0) {
         printInfo = true;
      }
      else if (strcmp(argv[i], "-samples") == 0 && i + 1 < argc) {
         i++;
         sample_count = sample_count_flag(strtol(argv[i], NULL, 10));
      }
      else if (strcmp(argv[i], "-present-mailbox") == 0) {
         desidered_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
      }
      else if (strcmp(argv[i], "-present-immediate") == 0) {
         desidered_present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      }
      else if (strcmp(argv[i], "-shader-object") == 0) {
         use_shader_object = true;
      }
      else if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
         i++;
         char *token;
         token = strtok(argv[i], "x");
         if (!token)
            continue;
         long tmp = strtol(token, NULL, 10);
         if (tmp > 0)
            width = tmp;
         if ((token = strtok(NULL, "x"))) {
            tmp = strtol(token, NULL, 10);
            if (tmp > 0)
               height = tmp;
         }
      }
      else if (strcmp(argv[i], "-fullscreen") == 0) {
         fullscreen = true;
      }
      else {
         usage();
         return -1;
      }
   }

   new_width = width, new_height = height;

   wsi = get_wsi_interface();
   wsi.set_wsi_callbacks(wsi_callbacks);

   wsi.init_display();
   wsi.init_window("vkgears", width, height, fullscreen);

   init_vk(wsi.required_extension_name);

   if (!check_sample_count_support(sample_count))
      error("Sample count not supported");

   if (printInfo)
      print_info();

   if (!wsi.create_surface(physical_device, instance, &surface))
      error("Failed to create surface!");

   configure_swapchain();
   create_swapchain();
   init_gears();

   bool first[4] = {false};

   while (1) {
      static int frames = 0;
      static double tRot0 = -1.0, tRate0 = -1.0;
      double dt, t = current_time();

      if (tRot0 < 0.0)
         tRot0 = t;
      dt = t - tRot0;
      tRot0 = t;

      if (animate) {
         /* advance rotation for next frame */
         angle += 70.0 * dt;  /* 70 degrees per second */
         if (angle > 3600.0)
            angle -= 3600.0;
      }

      if (wsi.update_window()) {
         printf("update window failed\n");
         break;
      }

      static uint32_t frame_index;
      assert(frame_index < ARRAY_SIZE(frame_data));
      vkWaitForFences(device, 1, &frame_data[frame_index].fence, VK_TRUE, UINT64_MAX);
      vkResetFences(device, 1, &frame_data[frame_index].fence);

      uint32_t image_index;
      VkResult result =
         vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                               frame_data[frame_index].semaphore, VK_NULL_HANDLE,
                               &image_index);
      if (result == VK_SUBOPTIMAL_KHR ||
          width != new_width || height != new_height) {
         recreate_swapchain();
         memset(first, 0, sizeof(first));
         continue;
      }
      assert(result == VK_SUCCESS);

      assert(image_index < ARRAY_SIZE(image_data));

      vkBeginCommandBuffer(frame_data[frame_index].cmd_buffer,
         &(VkCommandBufferBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0
         });

      /* projection matrix */
      float h = (float)height / width;
      struct ubo ubo;
      mat4_identity(ubo.projection);
      mat4_frustum_vk(ubo.projection, -1.0, 1.0, -h, +h, 5.0f, 60.0f);

      buffer_barrier(frame_data[frame_index].cmd_buffer,
         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT,
         0, 0,
         ubo_buffer, 0, sizeof(ubo));

      vkCmdUpdateBuffer(frame_data[frame_index].cmd_buffer, ubo_buffer, 0, sizeof(ubo), &ubo);

      buffer_barrier(frame_data[frame_index].cmd_buffer,
         VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
         VK_ACCESS_TRANSFER_WRITE_BIT,
         VK_ACCESS_UNIFORM_READ_BIT,
         ubo_buffer, 0, sizeof(ubo));

      vkCmdPipelineBarrier(frame_data[frame_index].cmd_buffer,
         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         0,
         0, NULL,
         0, NULL,
         1, &(VkImageMemoryBarrier) {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            NULL,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            !first[image_index] ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            0, 0,
            image_data[frame_index].image,
            .subresourceRange = {
               .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
               .baseMipLevel = 0,
               .levelCount = 1,
               .baseArrayLayer = 0,
               .layerCount = 1,
            },
         }
      );
      first[image_index] = true;

      vkCmdBeginRendering(frame_data[frame_index].cmd_buffer,
         &(VkRenderingInfo) {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { width, height } },
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = (VkRenderingAttachmentInfo[]) { {
               VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = sample_count != VK_SAMPLE_COUNT_1_BIT ? color_msaa_view : image_data[image_index].view,
               .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               .resolveMode = sample_count != VK_SAMPLE_COUNT_1_BIT ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
               .resolveImageView = sample_count != VK_SAMPLE_COUNT_1_BIT ? image_data[image_index].view : VK_NULL_HANDLE,
               .resolveImageLayout = sample_count != VK_SAMPLE_COUNT_1_BIT ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = sample_count != VK_SAMPLE_COUNT_1_BIT ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue.color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } },
            }},
            .pDepthAttachment = &(VkRenderingAttachmentInfo) {
               VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = depth_view,
               .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               .resolveMode = sample_count != VK_SAMPLE_COUNT_1_BIT ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
               .clearValue = { .depthStencil.depth = 1.0f },
            }
         });

      draw_gears(frame_data[frame_index].cmd_buffer);
      vkCmdEndRendering(frame_data[frame_index].cmd_buffer);
      vkCmdPipelineBarrier(frame_data[frame_index].cmd_buffer,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
         0,
         0, NULL,
         0, NULL,
         1, &(VkImageMemoryBarrier) {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            NULL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            0, 0,
            image_data[frame_index].image,
            .subresourceRange = {
               .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
               .baseMipLevel = 0,
               .levelCount = 1,
               .baseArrayLayer = 0,
               .layerCount = 1,
            },
         }
      );
      vkEndCommandBuffer(frame_data[frame_index].cmd_buffer);

      vkQueueSubmit(queue, 1,
         &(VkSubmitInfo) {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &frame_data[frame_index].semaphore,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &present_semaphore,
            .pWaitDstStageMask = (VkPipelineStageFlags []) {
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            },
            .commandBufferCount = 1,
            .pCommandBuffers = &frame_data[frame_index].cmd_buffer,
         }, frame_data[frame_index].fence);

      vkQueuePresentKHR(queue,
         &(VkPresentInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pWaitSemaphores = &present_semaphore,
            .waitSemaphoreCount = 1,
            .swapchainCount = 1,
            .pSwapchains = (VkSwapchainKHR[]) { swapchain, },
            .pImageIndices = (uint32_t[]) { image_index, },
            .pResults = &result,
         });

      frames++;

      frame_index++;
      if (frame_index == MAX_CONCURRENT_FRAMES)
         frame_index = 0;

      if (tRate0 < 0.0)
         tRate0 = t;
      if (t - tRate0 >= 5.0) {
         float seconds = t - tRate0;
         float fps = frames / seconds;
         printf("%d frames in %3.1f seconds = %6.3f FPS\n", frames, seconds,
               fps);
         fflush(stdout);
         tRate0 = t;
         frames = 0;
      }
   }

   wsi.fini_window();
   wsi.fini_display();
   return 0;
}
