//! iOS-only, non-FSR frame interpolation.
//!
//! This first implementation synthesizes a midpoint by blending consecutive world
//! frames. It is deliberately conservative: no optical-flow SDK, no FSR shaders and
//! no TAA dependency. Moving objects can show ghosting; this is a prototype intended
//! to validate the Metal presentation path before adding motion-compensated vectors.

use glam::{Mat4, Vec3};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

const FRAMES: usize = 6;

const BLEND_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var previous_frame: texture_2d<f32>;
@group(0) @binding(2) var frame_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.uv = uvs[index];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let before = textureSample(previous_frame, frame_sampler, in.uv);
    let after = textureSample(current_frame, frame_sampler, in.uv);
    return vec4<f32>(mix(before.rgb, after.rgb, 0.5), 1.0);
}
"#;

pub struct Camera {
    pub jittered: Mat4,
    pub view_proj: Mat4,
    pub depth: wgpu::TextureView,
    pub render_size: [u32; 2],
    pub jitter: [f32; 2],
    pub reproject: Option<Mat4>,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
}

struct Gpu {
    size: [u32; 2],
    world: wgpu::Texture,
    world_view: wgpu::TextureView,
    world_group: wgpu::BindGroup,
    previous: wgpu::Texture,
    previous_view: wgpu::TextureView,
    blend_layout: wgpu::BindGroupLayout,
    blend_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    previous_valid: bool,
}

pub struct FrameGen {
    pub on: bool,
    pub paused: bool,
    presenter: Option<Presenter>,
    offscreen_generated: Option<wgpu::Texture>,
    gpu: Option<Gpu>,
    last: Option<Instant>,
    interval: Duration,
    hand: Vec<crate::mesher::Vertex>,
    hand_view: Mat4,
    real: u32,
    made: u32,
    shown_since: Instant,
    pub made_share: Option<f32>,
}

impl FrameGen {
    pub fn new() -> Self {
        Self {
            on: false,
            paused: false,
            presenter: None,
            offscreen_generated: None,
            gpu: None,
            last: None,
            interval: Duration::from_millis(16),
            hand: Vec::new(),
            hand_view: Mat4::IDENTITY,
            real: 0,
            made: 0,
            shown_since: Instant::now(),
            made_share: None,
        }
    }

    pub fn stop(&mut self) {
        self.presenter = None;
        self.made_share = None;
        self.last = None;
        self.hand.clear();
        if let Some(gpu) = &mut self.gpu {
            gpu.previous_valid = false;
        }
        self.real = 0;
        self.made = 0;
    }

    pub fn halfway_hand(
        &mut self,
        hand: &[crate::mesher::Vertex],
        view: Mat4,
        out: &mut Vec<crate::mesher::Vertex>,
    ) {
        let same = self.hand.len() == hand.len()
            && self.hand.iter().zip(hand).all(|(a, b)| a.layer == b.layer && a.uv == b.uv);
        if same {
            let back = view.inverse();
            out.extend(self.hand.iter().zip(hand).map(|(before, now)| {
                let then = self.hand_view.transform_point3(before.pos.into());
                let mid = (then + view.transform_point3(now.pos.into())) * 0.5;
                crate::mesher::Vertex {
                    pos: back.transform_point3(mid).into(),
                    ..*now
                }
            }));
        } else {
            out.extend_from_slice(hand);
        }
        self.hand.clear();
        self.hand.extend_from_slice(hand);
        self.hand_view = view;
    }

    pub fn track_entities(
        &mut self,
        _device: &wgpu::Device,
        _atlas: &wgpu::TextureView,
        _entities: &[crate::mesher::Vertex],
        _spans: &[(u32, u64)],
        _camera: glam::DVec3,
    ) {
        // The current midpoint blend has no object-specific motion field.
    }

    pub fn camera_motion(&mut self, _view_proj: Mat4, _camera: glam::DVec3) -> Option<Mat4> {
        None
    }

    pub fn frames(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Option<&Arc<wgpu::Surface<'static>>>,
        offscreen: Option<&wgpu::Texture>,
        window: &Arc<winit::window::Window>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Option<(wgpu::Texture, wgpu::Texture)> {
        let Some(surface) = surface else {
            let real = offscreen.cloned().unwrap_or_else(|| frame(device, config));
            if self.offscreen_generated.as_ref().is_none_or(|t| t.size() != real.size()) {
                self.offscreen_generated = Some(frame(device, config));
            }
            return Some((real, self.offscreen_generated.as_ref()?.clone()));
        };

        if self.presenter.as_ref().is_some_and(|p| p.config != *config) {
            self.presenter = None;
        }
        let presenter = self.presenter.get_or_insert_with(|| {
            Presenter::start(device, queue, surface.clone(), window.clone(), config)
        });
        let pair = if presenter.failed() {
            None
        } else {
            presenter.free.recv().ok().zip(presenter.free.recv().ok())
        };
        if pair.is_none() {
            self.stop();
            self.paused = true;
        }
        pair
    }

    pub fn world(
        &mut self,
        device: &wgpu::Device,
        post: &super::world_post::Post,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = [config.width, config.height];
        if self.gpu.as_ref().is_none_or(|g| g.size != size) {
            let format = config.format.remove_srgb_suffix();
            let make_texture = |label: &'static str, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size[0].max(1),
                        height: size[1].max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let world = make_texture(
                "Horizon iOS frame generation current",
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let previous = make_texture(
                "Horizon iOS frame generation previous",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let world_view = world.create_view(&Default::default());
            let previous_view = previous.create_view(&Default::default());
            let blend_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Horizon iOS frame interpolation"),
                entries: &[
                    texture_entry(0),
                    texture_entry(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Horizon iOS frame interpolation"),
                source: wgpu::ShaderSource::Wgsl(BLEND_SHADER.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Horizon iOS frame interpolation"),
                bind_group_layouts: &[Some(&blend_layout)],
                ..Default::default()
            });
            let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Horizon iOS frame interpolation"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Horizon iOS frame interpolation"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let world_group = post.bind(device, &world_view);
            self.gpu = Some(Gpu {
                size,
                world,
                world_view,
                world_group,
                previous,
                previous_view,
                blend_layout,
                blend_pipeline,
                sampler,
                previous_valid: false,
            });
        }
        self.gpu.as_ref().expect("created above").world_view.clone()
    }

    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        post: &super::world_post::Post,
        screen: &wgpu::TextureView,
        generated: &wgpu::TextureView,
        _camera: &Camera,
    ) -> bool {
        let gpu = self.gpu.as_mut().expect("world() ran this frame");
        post.finish_with(encoder, screen, &gpu.world_group, None);
        let made = gpu.previous_valid;
        if made {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Horizon iOS interpolated frame"),
                layout: &gpu.blend_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu.world_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gpu.previous_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&gpu.sampler),
                    },
                ],
            });
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: generated,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizon iOS generated midpoint"),
                color_attachments: &attachments,
                ..Default::default()
            });
            pass.set_pipeline(&gpu.blend_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        } else {
            post.finish_with(encoder, generated, &gpu.world_group, None);
        }
        encoder.copy_texture_to_texture(
            gpu.world.as_image_copy(),
            gpu.previous.as_image_copy(),
            wgpu::Extent3d {
                width: gpu.size[0],
                height: gpu.size[1],
                depth_or_array_layers: 1,
            },
        );
        gpu.previous_valid = true;
        made
    }

    pub fn present(&mut self, real: wgpu::Texture, generated: wgpu::Texture, made: bool) {
        let now = Instant::now();
        if let Some(last) = self.last {
            let dt = (now - last).min(Duration::from_millis(250));
            self.interval = self.interval.mul_f32(0.9) + dt.mul_f32(0.1);
        }
        self.last = Some(now);
        self.real += 1;
        self.made += u32::from(made);
        if now - self.shown_since >= Duration::from_millis(500) {
            self.made_share = Some(self.made as f32 / self.real.max(1) as f32);
            self.real = 0;
            self.made = 0;
            self.shown_since = now;
        }
        let Some(send) = self.presenter.as_ref().and_then(|p| p.send.as_ref()) else {
            return;
        };
        let pair = if made {
            Pair {
                first: Some(generated),
                second: real,
                spare: None,
                interval: self.interval,
            }
        } else {
            Pair {
                first: None,
                second: real,
                spare: Some(generated),
                interval: self.interval,
            }
        };
        let _ = send.send(pair);
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn frame(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Horizon iOS frame generation frame"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &config.view_formats,
    })
}

struct Pair {
    first: Option<wgpu::Texture>,
    second: wgpu::Texture,
    spare: Option<wgpu::Texture>,
    interval: Duration,
}

struct Presenter {
    config: wgpu::SurfaceConfiguration,
    send: Option<mpsc::SyncSender<Pair>>,
    free: mpsc::Receiver<wgpu::Texture>,
    failed: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Presenter {
    fn start(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Arc<wgpu::Surface<'static>>,
        _window: Arc<winit::window::Window>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let (send, receive) = mpsc::sync_channel::<Pair>(1);
        let (give, free) = mpsc::channel();
        for _ in 0..FRAMES {
            let _ = give.send(frame(device, config));
        }
        let (device, queue) = (device.clone(), queue.clone());
        let failed = Arc::new(AtomicBool::new(false));
        let failed_here = failed.clone();
        let thread = thread::Builder::new()
            .name("Horizon iOS frame interpolation present".into())
            .spawn(move || {
                let show = |texture: &wgpu::Texture| {
                    if failed_here.load(Ordering::Relaxed) {
                        return;
                    }
                    let target = match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t) => t,
                        wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                            failed_here.store(true, Ordering::Relaxed);
                            t
                        }
                        _ => {
                            failed_here.store(true, Ordering::Relaxed);
                            return;
                        }
                    };
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_texture_to_texture(
                        texture.as_image_copy(),
                        target.texture.as_image_copy(),
                        target.texture.size(),
                    );
                    queue.submit([encoder.finish()]);
                    queue.present(target);
                };
                while let Ok(pair) = receive.recv() {
                    if let Some(first) = pair.first {
                        let shown = Instant::now();
                        show(&first);
                        let _ = give.send(first);
                        if let Some(wait) = (shown + pair.interval / 2)
                            .checked_duration_since(Instant::now())
                        {
                            thread::sleep(wait);
                        }
                    }
                    show(&pair.second);
                    let _ = give.send(pair.second);
                    if let Some(spare) = pair.spare {
                        let _ = give.send(spare);
                    }
                }
            })
            .expect("spawn iOS frame interpolation presenter");
        Self {
            config: config.clone(),
            send: Some(send),
            free,
            failed,
            thread: Some(thread),
        }
    }

    fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed) || self.thread.as_ref().is_none_or(|t| t.is_finished())
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.send = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
