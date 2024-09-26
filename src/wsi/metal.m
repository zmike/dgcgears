/*
 * Copyright 2024 Autodesk, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "wsi.h"

#include <vulkan/vulkan_metal.h>

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate> {
   NSWindow *_window;
@public
   CAMetalLayer *metal_layer;
}
@end

@interface WsiWindow : NSWindow <NSWindowDelegate> {}
@end

static struct wsi_callbacks wsi_callbacks;
static AppDelegate *app_delegate;

@implementation AppDelegate
- (void)init:(const char *)title
       withWidth:(NSInteger)width
      withHeight:(NSInteger)height
    isFullscreen:(BOOL)fullscreen
{
   NSRect frame = NSMakeRect(0, 0, width, height);

   _window = [[WsiWindow alloc]
      initWithContentRect:frame
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskResizable
                  backing:NSBackingStoreBuffered
                    defer:NO];
   _window.title = [NSString stringWithUTF8String:title];
   _window.delegate = app_delegate;

   [self setupMenuBar];

   NSView *view = _window.contentView;
   view.wantsLayer = YES;
   metal_layer = [CAMetalLayer layer];
   view.layer = metal_layer;

   [_window center];
   [_window makeKeyAndOrderFront:NSApp];

   /* run will block the thread, so we'll stop immediately in
      applicationDidFinishLaunching, and implement our own loop
      in update_window(). */
   [NSApp run];
}

- (void)setupMenuBar
{
   NSMenu *mainMenu = [[NSMenu alloc] initWithTitle:@""];
   [NSApp setMainMenu:mainMenu];

   NSMenuItem *applicationMenuItem = [mainMenu addItemWithTitle:@""
                                                         action:nil
                                                  keyEquivalent:@""];

   NSMenu *applicationMenu = [[NSMenu alloc] initWithTitle:@""];

   [applicationMenu
      addItemWithTitle:[@"Quit " stringByAppendingString:_window.title]
                action:@selector(terminate:)
         keyEquivalent:@"q"];

   [applicationMenuItem setSubmenu:applicationMenu];
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
   [NSApp stop:nil];
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)application
{
   wsi_callbacks.exit();
   return NSTerminateCancel;
}

- (void)windowDidResize:(NSNotification *)notification
{
   CGSize size = [[_window contentView] frame].size;
   wsi_callbacks.resize(size.width, size.height);
}
@end

@implementation WsiWindow
- (void)keyDown:(NSEvent *)event
{
   [WsiWindow keyPress:true
           withKeyCode:[event keyCode]];
}

- (void)keyUp:(NSEvent *)event
{
   [WsiWindow keyPress:false
           withKeyCode:[event keyCode]];
}

+ (void)keyPress:(bool)down
   withKeyCode:(int)keyCode
{
   switch (keyCode) {
   case 0x0:
      wsi_callbacks.key_press(down, WSI_KEY_A);
      break;
   case 0x35:
      wsi_callbacks.key_press(down, WSI_KEY_ESC);
      break;
   case 0x7E:
      wsi_callbacks.key_press(down, WSI_KEY_UP);
      break;
   case 0x7D:
      wsi_callbacks.key_press(down, WSI_KEY_DOWN);
      break;
   case 0x7B:
      wsi_callbacks.key_press(down, WSI_KEY_LEFT);
      break;
   case 0x7C:
      wsi_callbacks.key_press(down, WSI_KEY_RIGHT);
      break;
   }
}
@end

static void init_display()
{
   @autoreleasepool {
      [NSApplication sharedApplication];
      app_delegate = [AppDelegate alloc];
      [NSApp setDelegate:app_delegate];
      [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
   }
}

static void fini_display()
{
}

static void init_window(const char *title, int width, int height,
                        bool fullscreen)
{
   @autoreleasepool {
      [app_delegate init:title
               withWidth:width
            withHeight:height
            isFullscreen:fullscreen];
   }
}

static bool
update_window()
{
   @autoreleasepool {
      while (true) {
         NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                             untilDate:[NSDate distantPast]
                                                inMode:NSDefaultRunLoopMode
                                               dequeue:YES];

         if (event == nil) {
            break;
         }

         [NSApp sendEvent:event];
      }
   }

   return false;
}

static void
fini_window()
{
}

static void
set_wsi_callbacks(struct wsi_callbacks callbacks)
{
   wsi_callbacks = callbacks;
}

#define GET_INSTANCE_PROC(name) \
   PFN_##name name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);

static bool
create_surface(VkPhysicalDevice physical_device,
               VkInstance instance, VkSurfaceKHR *surface)
{
   GET_INSTANCE_PROC(vkCreateMetalSurfaceEXT)

   if (!vkCreateMetalSurfaceEXT) {
      fprintf(stderr, "Failed to load extension functions\n");
      return false;
   }

   return vkCreateMetalSurfaceEXT(instance,
                                 &(VkMetalSurfaceCreateInfoEXT) {
                                    .sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
                                    .pLayer = app_delegate->metal_layer,
                                 },
                                 NULL,
                                 surface) == VK_SUCCESS;
}

struct wsi_interface
metal_wsi_interface()
{
   return (struct wsi_interface) {
      .required_extension_name = VK_EXT_METAL_SURFACE_EXTENSION_NAME,

      .init_display = init_display,
      .fini_display = fini_display,

      .init_window = init_window,
      .update_window = update_window,
      .fini_window = fini_window,

      .set_wsi_callbacks = set_wsi_callbacks,

      .create_surface = create_surface,
   };
}
