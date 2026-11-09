//! The GPU tests' `wgpu::Instance`.

/// A new instance for one GPU test. On Windows it opens Vulkan and DirectX 12,
/// not OpenGL: WGL instances made and dropped from many test threads at once
/// deadlocked NVIDIA's driver, and the render tests hung most runs (#5238).
/// The game runs there on Vulkan or DirectX 12 unless neither works.
/// `WGPU_BACKEND` and the other `WGPU_*` variables still apply, as in the game.
pub(crate) fn instance() -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    if cfg!(windows) {
        descriptor.backends = wgpu::Backends::PRIMARY;
    }
    wgpu::Instance::new(descriptor.with_env())
}

/// The frame-generation GPU tests must exercise Metal on macOS CI, not silently skip.
#[cfg(target_os = "macos")]
#[test]
fn metal_adapter_is_available() {
    let adapter = pollster::block_on(instance().request_adapter(&Default::default()))
        .expect("macOS Metal adapter is required for frame-generation GPU tests");
    assert_eq!(
        adapter.get_info().backend,
        wgpu::Backend::Metal,
        "frame-generation CI must run against the Metal backend",
    );
}

/// Construct every world-post pipeline on Metal so WGSL and pipeline-layout
/// validation for spatial upscaling is exercised in CI, not deferred to launch.
#[cfg(target_os = "macos")]
#[test]
fn metal_world_post_pipelines_compile() {
    let adapter = pollster::block_on(instance().request_adapter(&Default::default()))
        .expect("macOS Metal adapter is required for world-post pipeline tests");
    assert_eq!(
        adapter.get_info().backend,
        wgpu::Backend::Metal,
        "world-post CI must run against the Metal backend",
    );

    let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))
        .expect("a default Metal device must be available for world-post pipeline tests");
    let _post = super::world_post::Post::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
}
