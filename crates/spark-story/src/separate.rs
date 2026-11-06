//! Instruction tokens versus situation tokens in the final residual stream.
//!
//! The input embedding table is frozen, so its rows cannot show whether LoRA
//! changed instruction following. This probe reads the residual at each token
//! after the last norm. The plotted plane is the one that separates the
//! format groups, not the highest-variance plane of every token. With a LoRA
//! file, the first panel is how each group's mean moved. The second panel is
//! a neighborhood map of the tangent displacement, so tokens that moved the
//! same way stay together.

use std::fs;
use std::io::Write;
use std::path::Path;

use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use rollout_engine::SparkTokenizer;
use spark_burn::{
    LoraSpec, SparkConfig, SparkModel, attach_lora, bf16_device, load_lora_adapters,
    load_spark_checkpoint, sidecar_dir,
};

use crate::seeds::{self, SeedKind};

const SITUATIONS: &[&str] = &[
    "A baker locks the alley door as the oven timer and the rain start together.",
    "Two ferry ropes slap the dock while a child counts orange crates missing from the manifest.",
    "The clinic fridge hums louder than the hallway, and a nurse keeps one hand on a warm glass vial.",
    "Limestone dust coats the yard as the saw stops halfway through a block ordered for a stair.",
    "A tram depot bell rings twice, and the mechanic leaves a brass key on the wet running board.",
    "Honey boxes buzz under a late sun while someone mends a split frame with borrowed wire.",
    "Parcel tags spill across the station table after the noon train leaves one cage unlocked.",
    "Greenhouse glass ticks in a downpour, and a tomato vine leans onto a neighbor's labeled pot.",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Instruction,
    Context,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Series {
    Instruction(SeedKind),
    Context,
}

struct Spans {
    instruction: Vec<(usize, usize)>,
    context: (usize, usize),
}

struct Cloud {
    instruction: Vec<(SeedKind, Vec<f32>)>,
    context: Vec<Vec<f32>>,
}

struct Projection {
    coords: Vec<(f64, f64, Series)>,
    x_label: String,
    y_label: String,
    /// Shared limits for panels that must be read on one scale.
    frame: Option<(f64, f64, f64, f64)>,
    /// Segment from a group mean to where that mean moved, in plot coordinates.
    arrows: Vec<(f64, f64, f64, f64, Series)>,
    /// Draw the y = 0 line. The comparison scatters put the LoRA displacement on y.
    baseline: bool,
    span_cols: usize,
}

struct Basis {
    mean: Vec<f64>,
    axes: [Vec<f64>; 2],
    shares: [f64; 2],
}

pub struct SeparationReport {
    pub name: String,
    pub instruction_tokens: usize,
    pub context_tokens: usize,
    pub instruction_rms: f64,
    pub context_rms: f64,
    pub tightness: f64,
    pub centroid_gap: f64,
}

pub fn instruction_space(
    checkpoint: &Path,
    lora: Option<&Path>,
    kind: Option<SeedKind>,
    situations: usize,
    out: &Path,
) -> Result<Vec<SeparationReport>, String> {
    let count = situations.clamp(2, SITUATIONS.len());
    let kinds = kind
        .map(|kind| vec![kind])
        .unwrap_or_else(|| SeedKind::ALL.to_vec());
    let directory = sidecar_dir(checkpoint);
    let config = SparkConfig::load_sidecar(checkpoint)?;
    let tokenizer = SparkTokenizer::open(&directory.join("tokenizer.json"))?;
    let device = bf16_device(false)?;
    let formats = match kind {
        Some(kind) => kind.label().to_string(),
        None => "all formats".to_string(),
    };
    println!(
        "reading residuals for {formats}, {count} situations from {}",
        checkpoint.display()
    );
    let base = SparkModel::new(&config, &device);
    let base = load_spark_checkpoint(base, checkpoint).map_err(|err| err.to_string())?;
    let base = base.valid();
    let mut reports = Vec::new();
    let base_cloud = collect(&base, &tokenizer, &kinds, count)?;
    let base_report = report("base checkpoint", &base_cloud)?;
    print_report(&base_report);
    print_kind_radii(&kinds, &base_cloud);
    reports.push(base_report);
    let panels = if let Some(lora) = lora {
        println!("reading residuals with LoRA {}", lora.display());
        let adapted = SparkModel::new(&config, &device);
        let adapted = load_spark_checkpoint(adapted, checkpoint).map_err(|err| err.to_string())?;
        let adapted = attach_lora(adapted, LoraSpec::attention_default())?;
        let adapted = load_lora_adapters(adapted, lora)?.valid();
        let cloud = collect(&adapted, &tokenizer, &kinds, count)?;
        let adapted_report = report("attention LoRA", &cloud)?;
        print_report(&adapted_report);
        print_kind_radii(&kinds, &cloud);
        print_movement(&base_cloud, &cloud)?;
        reports.push(adapted_report);
        paired_panels(&base_cloud, &cloud)?
    } else {
        panels_for("base checkpoint", &base_cloud)?
    };
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut file = fs::File::create(out).map_err(|err| err.to_string())?;
    file.write_all(render_svg(&panels).as_bytes())
        .map_err(|err| err.to_string())?;
    println!("wrote {}", out.display());
    Ok(reports)
}

fn collect(
    model: &SparkModel,
    tokenizer: &SparkTokenizer,
    kinds: &[SeedKind],
    situations: usize,
) -> Result<Cloud, String> {
    let mut instruction = Vec::new();
    let mut context = Vec::new();
    for kind in kinds {
        for (index, situation) in SITUATIONS.iter().take(situations).enumerate() {
            let prompt = seeds::seed(index, *kind, *situation).prompt;
            let (ids, offsets) = tokenizer.encode_offsets(&prompt)?;
            let spans = byte_spans(&prompt, *kind, situation)?;
            let device = model.embed.weight.val().device();
            let tokens = ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
            let input =
                Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, ids.len()]), &device);
            let hidden = host_f32(model.residual_states(input));
            let width = hidden.len() / ids.len();
            for (token, offset) in offsets.iter().enumerate() {
                let row = hidden[token * width..(token + 1) * width].to_vec();
                match role_at(*offset, &spans) {
                    Some(Role::Instruction) => instruction.push((*kind, row)),
                    Some(Role::Context) => context.push(row),
                    None => {}
                }
            }
        }
    }
    if instruction.len() < 2 || context.len() < 2 {
        return Err(
            "the probe needs at least two instruction tokens and two situation tokens".into(),
        );
    }
    Ok(Cloud {
        instruction,
        context,
    })
}

fn byte_spans(prompt: &str, kind: SeedKind, situation: &str) -> Result<Spans, String> {
    let instruction = seeds::instruction(kind);
    let instruction_at = prompt
        .find(instruction)
        .ok_or("the format instruction is missing from the prompt")?;
    let close = "Stay inside this format for the whole piece.";
    let close_at = prompt
        .find(close)
        .ok_or("the format reminder is missing from the prompt")?;
    let marker = "Situation:\n";
    let context_at = prompt
        .find(marker)
        .ok_or("the situation marker is missing from the prompt")?
        + marker.len();
    let context = &prompt[context_at..];
    if !context.starts_with(situation) {
        return Err("the situation text is not where the prompt says it is".into());
    }
    Ok(Spans {
        instruction: vec![
            (instruction_at, instruction_at + instruction.len()),
            (close_at, close_at + close.len()),
        ],
        context: (context_at, context_at + situation.len()),
    })
}

fn role_at(offset: (usize, usize), spans: &Spans) -> Option<Role> {
    let width = offset.1.saturating_sub(offset.0);
    if width == 0 {
        return None;
    }
    let instruction = spans
        .instruction
        .iter()
        .map(|span| overlap(offset, *span))
        .max()
        .unwrap_or(0);
    let context = overlap(offset, spans.context);
    if instruction == 0 && context == 0 {
        return None;
    }
    if instruction >= context && instruction * 2 >= width {
        Some(Role::Instruction)
    } else if context * 2 >= width {
        Some(Role::Context)
    } else {
        None
    }
}

fn overlap(token: (usize, usize), span: (usize, usize)) -> usize {
    let start = token.0.max(span.0);
    let end = token.1.min(span.1);
    end.saturating_sub(start)
}

fn instruction_rows(cloud: &Cloud) -> Vec<Vec<f32>> {
    cloud
        .instruction
        .iter()
        .map(|(_, row)| row.clone())
        .collect()
}

fn report(name: &str, cloud: &Cloud) -> Result<SeparationReport, String> {
    let instruction = instruction_rows(cloud);
    let instruction_rms = rms_radius(&instruction);
    let context_rms = rms_radius(&cloud.context);
    if instruction_rms < 1e-8 {
        return Err("instruction residuals have no spread".into());
    }
    Ok(SeparationReport {
        name: name.to_string(),
        instruction_tokens: cloud.instruction.len(),
        context_tokens: cloud.context.len(),
        instruction_rms,
        context_rms,
        tightness: context_rms / instruction_rms,
        centroid_gap: centroid_distance(&instruction, &cloud.context),
    })
}

fn print_kind_radii(kinds: &[SeedKind], cloud: &Cloud) {
    if kinds.len() < 2 {
        return;
    }
    for kind in kinds {
        let rows = cloud
            .instruction
            .iter()
            .filter(|(row_kind, _)| row_kind == kind)
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        if rows.len() < 2 {
            continue;
        }
        println!(
            "  {}: instruction tokens {} rms {:.4}",
            kind.label(),
            rows.len(),
            rms_radius(&rows)
        );
    }
}

fn print_report(report: &SeparationReport) {
    println!(
        "{}: instruction tokens {} rms {:.4}, situation tokens {} rms {:.4}, tightness {:.3}, centroid gap {:.4}",
        report.name,
        report.instruction_tokens,
        report.instruction_rms,
        report.context_tokens,
        report.context_rms,
        report.tightness,
        report.centroid_gap
    );
    if report.tightness > 1.0 {
        println!(
            "{} instruction residuals sit in a tighter cloud than the situation residuals",
            report.name
        );
    } else {
        println!(
            "{} situation residuals are at least as tight as the instruction residuals",
            report.name
        );
    }
}

fn panels_for(name: &str, cloud: &Cloud) -> Result<Vec<Projection>, String> {
    let points = pooled(cloud);
    let (group, y_within) = group_plane(&points)?;
    let separating = separating_plane(&points)?;
    let (x_label, y_label) = group_axis_labels(&group, y_within);
    Ok(vec![
        Projection {
            coords: assign(&project_rows(&points, &group), &points),
            x_label,
            y_label,
            frame: None,
            arrows: Vec::new(),
            baseline: false,
            span_cols: 1,
        },
        Projection {
            coords: separating,
            x_label: format!("{name}: instruction mean minus situation mean"),
            y_label: "Leading direction orthogonal to that axis".to_string(),
            frame: None,
            arrows: Vec::new(),
            baseline: false,
            span_cols: 1,
        },
    ])
}

fn paired_panels(base: &Cloud, adapted: &Cloud) -> Result<Vec<Projection>, String> {
    let base_points = pooled(base);
    let adapted_points = pooled(adapted);
    if base_points.len() != adapted_points.len()
        || base_points
            .iter()
            .zip(&adapted_points)
            .any(|((_, left), (_, right))| left != right)
    {
        return Err("base and LoRA residuals are not paired token for token".into());
    }
    let shifts = group_shifts(&base_points, &adapted_points)?;
    let movement = shift_plane(&shifts)?;
    let shift_rows = shifts
        .iter()
        .map(|(_, row)| row.clone())
        .collect::<Vec<_>>();
    let shift_scores = project(&shift_rows, &movement);
    let mut shift_coords = Vec::new();
    let mut arrows = Vec::new();
    for (score, (series, _)) in shift_scores.iter().zip(&shifts) {
        shift_coords.push((score[0], score[1], *series));
        arrows.push((0.0, 0.0, score[0], score[1], *series));
    }
    let (x0, x1, y0, y1) = bounds(&shift_coords);
    let shift_frame = (x0.min(0.0), x1.max(0.0), y0.min(0.0), y1.max(0.0));

    let base_rows: Vec<Vec<f32>> = base_points.iter().map(|(r, _)| r.clone()).collect();
    let adapted_rows: Vec<Vec<f32>> = adapted_points.iter().map(|(r, _)| r.clone()).collect();
    let dim = base_rows[0].len();

    let mut base_center = vec![0.0_f64; dim];
    for row in &base_rows {
        for (s, v) in base_center.iter_mut().zip(row) {
            *s += f64::from(*v);
        }
    }
    for v in &mut base_center {
        *v /= base_rows.len() as f64;
    }

    // Top Panel: 2 slots wide, showing the existing shift of each group mean plot
    let top_panel = Projection {
        coords: shift_coords,
        x_label: format!(
            "Shift of each group mean, axis 1 ({:.0}% of that shift)",
            movement.shares[0] * 100.0
        ),
        y_label: format!(
            "Axis 2 of the group-mean shift ({:.0}%)",
            movement.shares[1] * 100.0
        ),
        frame: Some(shift_frame),
        arrows,
        baseline: false,
        span_cols: 2,
    };

    // Option A: Shift subspace before and after LoRA
    let (before_a, after_a) = option_a_pair(
        &base_rows,
        &adapted_rows,
        &base_points,
        &adapted_points,
        &movement,
        &base_center,
    );

    // Option B: Supervised Discriminant before and after LoRA
    let (before_b, after_b) = option_b_pair(
        &base_rows,
        &adapted_rows,
        &base_points,
        &adapted_points,
        dim,
    )?;

    // Option C: Directional Phase Space before and after LoRA
    let (before_c, after_c) =
        option_c_pair(&base_points, &adapted_points, &shifts, &base_center, dim)?;

    Ok(vec![
        top_panel, before_a, after_a, before_b, after_b, before_c, after_c,
    ])
}

fn option_a_pair(
    base_rows: &[Vec<f32>],
    adapted_rows: &[Vec<f32>],
    base_points: &[(Vec<f32>, Series)],
    adapted_points: &[(Vec<f32>, Series)],
    movement: &Basis,
    base_center: &[f64],
) -> (Projection, Projection) {
    let basis_a = Basis {
        axes: movement.axes.clone(),
        mean: base_center.to_vec(),
        shares: movement.shares,
    };
    let before_scores = project(base_rows, &basis_a);
    let after_scores = project(adapted_rows, &basis_a);

    let before_coords = assign(&before_scores, base_points);
    let after_coords = assign(&after_scores, adapted_points);

    let (x0_1, x1_1, y0_1, y1_1) = bounds(&before_coords);
    let (x0_2, x1_2, y0_2, y1_2) = bounds(&after_coords);
    let frame = Some((
        x0_1.min(x0_2),
        x1_1.max(x1_2),
        y0_1.min(y0_2),
        y1_1.max(y1_2),
    ));

    let y_lbl = format!(
        "Shift axis 1 ({:.0}%) vs axis 2 ({:.0}%)",
        movement.shares[0] * 100.0,
        movement.shares[1] * 100.0
    );

    (
        Projection {
            coords: before_coords,
            x_label: "Option A: Shift subspace (Before LoRA)".to_string(),
            y_label: y_lbl.clone(),
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
        Projection {
            coords: after_coords,
            x_label: "Option A: Shift subspace (With LoRA)".to_string(),
            y_label: y_lbl,
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
    )
}

fn option_b_pair(
    base_rows: &[Vec<f32>],
    adapted_rows: &[Vec<f32>],
    base_points: &[(Vec<f32>, Series)],
    adapted_points: &[(Vec<f32>, Series)],
    dim: usize,
) -> Result<(Projection, Projection), String> {
    let mut instr_mean = vec![0.0_f64; dim];
    let mut instr_count = 0.0_f64;
    let mut ctx_mean = vec![0.0_f64; dim];
    let mut ctx_count = 0.0_f64;
    for (row, series) in adapted_points {
        match series {
            Series::Instruction(_) => {
                instr_count += 1.0;
                for (s, v) in instr_mean.iter_mut().zip(row) {
                    *s += f64::from(*v);
                }
            }
            Series::Context => {
                ctx_count += 1.0;
                for (s, v) in ctx_mean.iter_mut().zip(row) {
                    *s += f64::from(*v);
                }
            }
        }
    }
    for v in &mut instr_mean {
        *v /= instr_count.max(1.0);
    }
    for v in &mut ctx_mean {
        *v /= ctx_count.max(1.0);
    }

    let mut axis_1: Vec<f64> = instr_mean
        .iter()
        .zip(&ctx_mean)
        .map(|(i, c)| i - c)
        .collect();
    let norm_1 = l2(&axis_1).max(1e-12);
    for v in &mut axis_1 {
        *v /= norm_1;
    }

    let mut format_deviations = Vec::new();
    for series in series_in_points(adapted_points) {
        if matches!(series, Series::Instruction(_)) {
            let mean = mean_row(adapted_points, series);
            let along = axis_1.iter().zip(&mean).map(|(a, b)| a * b).sum::<f64>();
            let mut dev = vec![0.0_f64; dim];
            for (idx, v) in mean.iter().enumerate() {
                dev[idx] = v - along * axis_1[idx];
            }
            format_deviations.push(dev);
        }
    }
    let axis_2 = if format_deviations.len() >= 2 {
        match basis_from_deviations(&format_deviations, vec![0.0; dim]) {
            Ok(basis) => basis.axes[0].clone(),
            Err(_) => orthogonal_vector(&axis_1),
        }
    } else {
        orthogonal_vector(&axis_1)
    };

    let center_b: Vec<f64> = instr_mean
        .iter()
        .zip(&ctx_mean)
        .map(|(i, c)| (i + c) * 0.5)
        .collect();
    let basis_b = Basis {
        axes: [axis_1, axis_2],
        mean: center_b,
        shares: [1.0, 1.0],
    };

    let before_scores = project(base_rows, &basis_b);
    let after_scores = project(adapted_rows, &basis_b);

    let before_coords = assign(&before_scores, base_points);
    let after_coords = assign(&after_scores, adapted_points);

    let (x0_1, x1_1, y0_1, y1_1) = bounds(&before_coords);
    let (x0_2, x1_2, y0_2, y1_2) = bounds(&after_coords);
    let frame = Some((
        x0_1.min(x0_2),
        x1_1.max(x1_2),
        y0_1.min(y0_2),
        y1_1.max(y1_2),
    ));

    let y_lbl = "x = Instruction vs situation, y = Format spread".to_string();

    Ok((
        Projection {
            coords: before_coords,
            x_label: "Option B: Discriminant axis (Before LoRA)".to_string(),
            y_label: y_lbl.clone(),
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
        Projection {
            coords: after_coords,
            x_label: "Option B: Discriminant axis (With LoRA)".to_string(),
            y_label: y_lbl,
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
    ))
}

fn orthogonal_vector(axis: &[f64]) -> Vec<f64> {
    let dim = axis.len();
    let mut other = vec![0.0; dim];
    let min_idx = axis
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);
    other[min_idx] = 1.0;
    let along: f64 = axis.iter().zip(&other).map(|(a, b)| a * b).sum();
    for (s, v) in other.iter_mut().zip(axis) {
        *s -= along * v;
    }
    normalize(&other)
}

fn option_c_pair(
    base_points: &[(Vec<f32>, Series)],
    adapted_points: &[(Vec<f32>, Series)],
    shifts: &[(Series, Vec<f32>)],
    base_center: &[f64],
    dim: usize,
) -> Result<(Projection, Projection), String> {
    let mut instr_shift = vec![0.0_f64; dim];
    let mut ctx_shift = vec![0.0_f64; dim];
    for (series, shift) in shifts {
        match series {
            Series::Instruction(_) => {
                for (slot, value) in instr_shift.iter_mut().zip(shift) {
                    *slot += f64::from(*value);
                }
            }
            Series::Context => {
                for (slot, value) in ctx_shift.iter_mut().zip(shift) {
                    *slot += f64::from(*value);
                }
            }
        }
    }
    let norm_instr = l2(&instr_shift).max(1e-12);
    for v in &mut instr_shift {
        *v /= norm_instr;
    }
    let norm_ctx = l2(&ctx_shift).max(1e-12);
    for v in &mut ctx_shift {
        *v /= norm_ctx;
    }

    let before_coords: Vec<(f64, f64, Series)> = base_points
        .iter()
        .map(|(row, series)| {
            let centered: Vec<f64> = row
                .iter()
                .zip(base_center)
                .map(|(r, c)| f64::from(*r) - c)
                .collect();
            let x: f64 = centered.iter().zip(&instr_shift).map(|(a, b)| a * b).sum();
            let y: f64 = centered.iter().zip(&ctx_shift).map(|(a, b)| a * b).sum();
            (x, y, *series)
        })
        .collect();

    let after_coords: Vec<(f64, f64, Series)> = adapted_points
        .iter()
        .map(|(row, series)| {
            let centered: Vec<f64> = row
                .iter()
                .zip(base_center)
                .map(|(r, c)| f64::from(*r) - c)
                .collect();
            let x: f64 = centered.iter().zip(&instr_shift).map(|(a, b)| a * b).sum();
            let y: f64 = centered.iter().zip(&ctx_shift).map(|(a, b)| a * b).sum();
            (x, y, *series)
        })
        .collect();

    let (x0_1, x1_1, y0_1, y1_1) = bounds(&before_coords);
    let (x0_2, x1_2, y0_2, y1_2) = bounds(&after_coords);
    let frame = Some((
        x0_1.min(x0_2),
        x1_1.max(x1_2),
        y0_1.min(y0_2),
        y1_1.max(y1_2),
    ));

    let y_lbl = "x = Instruction steering, y = Situation steering".to_string();

    Ok((
        Projection {
            coords: before_coords,
            x_label: "Option C: Directional phase space (Before LoRA)".to_string(),
            y_label: y_lbl.clone(),
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
        Projection {
            coords: after_coords,
            x_label: "Option C: Directional phase space (With LoRA)".to_string(),
            y_label: y_lbl,
            frame,
            arrows: Vec::new(),
            baseline: true,
            span_cols: 1,
        },
    ))
}

/// LoRA change with the radial piece removed.
///
/// The final residual is normalized, so a change of length is not a change of
/// direction. The tangent piece is the part orthogonal to the base residual.
fn tangent_displacements(
    base: &[(Vec<f32>, Series)],
    adapted: &[(Vec<f32>, Series)],
) -> Vec<(Vec<f32>, Series)> {
    base.iter()
        .zip(adapted)
        .map(|((before, series), (after, _))| {
            let mut delta = Vec::with_capacity(before.len());
            let mut along = 0.0;
            let mut norm = 0.0;
            for (left, right) in before.iter().zip(after) {
                let change = f64::from(*right) - f64::from(*left);
                delta.push(change);
                along += change * f64::from(*left);
                norm += f64::from(*left) * f64::from(*left);
            }
            if norm > 1e-8 {
                let scale = along / norm;
                for (slot, value) in delta.iter_mut().zip(before) {
                    *slot -= scale * f64::from(*value);
                }
            }
            let row = delta.into_iter().map(|value| value as f32).collect();
            (row, *series)
        })
        .collect()
}

/// Laplacian eigenmap of the displacement vectors.
///
/// A global axis mixes tokens whose moves differ in several directions. This
/// keeps each token next to the tokens whose LoRA displacement is nearest,
/// then draws the two smoothest coordinates of that neighborhood graph.
fn neighborhood_map(points: &[(Vec<f32>, Series)]) -> Result<Vec<(f64, f64, Series)>, String> {
    if points.len() < 4 {
        return Err("a neighborhood map needs at least four tokens".into());
    }
    let rows = random_projection(points);
    let k = if points.len() < 40 {
        (points.len() / 3).clamp(2, points.len() - 1)
    } else {
        15.min(points.len() - 1)
    };
    let graph = neighborhood_weights(&rows, k)?;
    let embedding = smoothest_axes(&graph)?;
    Ok(points
        .iter()
        .zip(embedding)
        .map(|((_, series), score)| (score[0], score[1], *series))
        .collect())
}

fn random_projection(points: &[(Vec<f32>, Series)]) -> Vec<Vec<f64>> {
    const TARGET: usize = 32;
    let dim = points[0].0.len();
    if dim <= TARGET {
        return points
            .iter()
            .map(|(row, _)| row.iter().map(|value| f64::from(*value)).collect())
            .collect();
    }
    let mut seed = 0x5a17_u64;
    let mut signs = vec![0.0_f64; TARGET * dim];
    let scale = (dim as f64).sqrt();
    for sign in &mut signs {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *sign = if seed & 1 == 0 { -1.0 } else { 1.0 } / scale;
    }
    points
        .iter()
        .map(|(row, _)| {
            let mut projected = vec![0.0; TARGET];
            for (index, value) in row.iter().enumerate() {
                let value = f64::from(*value);
                for axis in 0..TARGET {
                    projected[axis] += value * signs[axis * dim + index];
                }
            }
            projected
        })
        .collect()
}

fn neighborhood_weights(rows: &[Vec<f64>], k: usize) -> Result<Vec<Vec<f64>>, String> {
    let n = rows.len();
    let mut neighbor_dist = Vec::new();
    let mut lists = Vec::with_capacity(n);
    for i in 0..n {
        let mut nearest = Vec::with_capacity(k);
        for j in 0..n {
            if i == j {
                continue;
            }
            let distance = squared_distance(&rows[i], &rows[j]);
            if nearest.len() < k {
                nearest.push((j, distance));
            } else if distance < nearest[k - 1].1 {
                nearest[k - 1] = (j, distance);
            } else {
                continue;
            }
            nearest.sort_by(|left, right| left.1.total_cmp(&right.1));
        }
        neighbor_dist.extend(nearest.iter().map(|(_, distance)| *distance));
        lists.push(nearest);
    }
    neighbor_dist.retain(|distance| *distance > 1e-18);
    neighbor_dist.sort_by(|left, right| left.total_cmp(right));
    let bandwidth = neighbor_dist
        .get(neighbor_dist.len() / 2)
        .copied()
        .unwrap_or(1.0);
    let mut weights = vec![vec![0.0_f64; n]; n];
    for (i, nearest) in lists.iter().enumerate() {
        for (j, distance) in nearest {
            let weight = (-distance / bandwidth).exp();
            weights[i][*j] = weights[i][*j].max(weight);
            weights[*j][i] = weights[*j][i].max(weight);
        }
    }
    Ok(weights)
}

fn smoothest_axes(weights: &[Vec<f64>]) -> Result<Vec<[f64; 2]>, String> {
    let n = weights.len();
    let mut degree: Vec<f64> = weights.iter().map(|row| row.iter().sum()).collect();
    for value in &mut degree {
        if *value < 1e-12 {
            *value = 1.0;
        }
    }
    let mut axes = Vec::new();
    axes.push(normalize(
        &degree.iter().map(|value| value.sqrt()).collect::<Vec<_>>(),
    ));
    for _ in 0..2 {
        let start: Vec<f64> = (0..n)
            .map(|index| ((index * 17) % 11) as f64 - 5.0)
            .collect();
        axes.push(largest_axis(weights, &degree, &start, &axes));
    }
    Ok((0..n)
        .map(|index| [axes[1][index], axes[2][index]])
        .collect())
}

fn largest_axis(
    weights: &[Vec<f64>],
    degree: &[f64],
    start: &[f64],
    forbidden: &[Vec<f64>],
) -> Vec<f64> {
    let mut vector = normalize(start);
    for _ in 0..64 {
        let mut next = affinity_apply(weights, degree, &vector);
        for axis in forbidden {
            let along = next
                .iter()
                .zip(axis)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            for (slot, direction) in next.iter_mut().zip(axis) {
                *slot -= along * direction;
            }
        }
        let norm = l2(&next);
        if norm < 1e-15 {
            break;
        }
        for value in &mut next {
            *value /= norm;
        }
        vector = next;
    }
    vector
}

fn affinity_apply(weights: &[Vec<f64>], degree: &[f64], vector: &[f64]) -> Vec<f64> {
    let scaled = vector
        .iter()
        .zip(degree)
        .map(|(value, weight)| value / weight.sqrt())
        .collect::<Vec<_>>();
    weights
        .iter()
        .zip(degree)
        .map(|(row, weight)| {
            row.iter()
                .zip(&scaled)
                .map(|(edge, value)| edge * value)
                .sum::<f64>()
                / weight.sqrt()
        })
        .collect()
}

fn normalize(vector: &[f64]) -> Vec<f64> {
    let norm = l2(vector).max(1e-12);
    vector.iter().map(|value| value / norm).collect()
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn centroid_of(coords: &[(f64, f64, Series)], series: Series) -> Option<(f64, f64)> {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut count = 0.0;
    for (px, py, item) in coords {
        if *item != series {
            continue;
        }
        x += px;
        y += py;
        count += 1.0;
    }
    if count == 0.0 {
        None
    } else {
        Some((x / count, y / count))
    }
}

fn group_axis_labels(basis: &Basis, y_within: bool) -> (String, String) {
    let x_label = format!(
        "Group axis 1 ({:.0}% of the separation among groups)",
        basis.shares[0] * 100.0
    );
    let y_label = if y_within {
        "Leading spread inside the groups".to_string()
    } else {
        format!(
            "Group axis 2 ({:.0}% of the separation among groups)",
            basis.shares[1] * 100.0
        )
    };
    (x_label, y_label)
}

fn group_shifts(
    base: &[(Vec<f32>, Series)],
    adapted: &[(Vec<f32>, Series)],
) -> Result<Vec<(Series, Vec<f32>)>, String> {
    let mut shifts = Vec::new();
    for series in series_in_points(base) {
        let before = mean_row(base, series);
        let after = mean_row(adapted, series);
        if before.is_empty() || after.is_empty() {
            return Err("a group is missing from the LoRA residuals".into());
        }
        let delta = before
            .iter()
            .zip(&after)
            .map(|(left, right)| (right - left) as f32)
            .collect();
        shifts.push((series, delta));
    }
    if shifts.len() < 2 {
        return Err("the shift plot needs at least two groups".into());
    }
    Ok(shifts)
}

fn mean_row(points: &[(Vec<f32>, Series)], series: Series) -> Vec<f64> {
    let mut mean = Vec::new();
    let mut count = 0.0_f64;
    for (row, item) in points {
        if *item != series {
            continue;
        }
        if mean.is_empty() {
            mean = vec![0.0; row.len()];
        }
        count += 1.0;
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    for value in &mut mean {
        *value /= count.max(1.0);
    }
    mean
}

fn print_movement(base: &Cloud, adapted: &Cloud) -> Result<(), String> {
    let instruction = relative_movement(&instruction_rows(base), &instruction_rows(adapted))?;
    let context = relative_movement(&base.context, &adapted.context)?;
    println!(
        "residual movement: instruction {instruction:.4} of its norm, situation {context:.4} of its norm"
    );
    if instruction > context {
        println!("the adapter moved instruction residuals farther than situation residuals");
    } else {
        println!("situation residuals moved at least as far as instruction residuals");
    }
    Ok(())
}

fn relative_movement(before: &[Vec<f32>], after: &[Vec<f32>]) -> Result<f64, String> {
    if before.len() != after.len() || before.is_empty() {
        return Err("residual pairs do not line up".into());
    }
    let mut delta = 0.0;
    let mut scale = 0.0;
    for (left, right) in before.iter().zip(after) {
        for (before, after) in left.iter().zip(right) {
            let change = f64::from(*after - *before);
            delta += change * change;
            let value = f64::from(*before);
            scale += value * value;
        }
    }
    Ok((delta / scale.max(1e-12)).sqrt())
}

fn assign(scores: &[[f64; 2]], points: &[(Vec<f32>, Series)]) -> Vec<(f64, f64, Series)> {
    scores
        .iter()
        .zip(points)
        .map(|(score, (_, series))| (score[0], score[1], *series))
        .collect()
}

fn pooled(cloud: &Cloud) -> Vec<(Vec<f32>, Series)> {
    cloud
        .instruction
        .iter()
        .cloned()
        .map(|(kind, row)| (row, Series::Instruction(kind)))
        .chain(
            cloud
                .context
                .iter()
                .cloned()
                .map(|row| (row, Series::Context)),
        )
        .collect()
}

fn series_in_points(points: &[(Vec<f32>, Series)]) -> Vec<Series> {
    let mut found = Vec::new();
    for kind in SeedKind::ALL {
        let series = Series::Instruction(kind);
        if points.iter().any(|(_, item)| *item == series) {
            found.push(series);
        }
    }
    if points.iter().any(|(_, item)| *item == Series::Context) {
        found.push(Series::Context);
    }
    found
}

fn project_rows(points: &[(Vec<f32>, Series)], basis: &Basis) -> Vec<[f64; 2]> {
    let rows = points
        .iter()
        .map(|(row, _)| row.clone())
        .collect::<Vec<_>>();
    project(&rows, basis)
}

/// Axes that hold the separation among group means.
///
/// Every token still has its own residual, and that token-to-token scatter
/// is what made a variance PCA report about 7% and 6%. The difference among
/// the group means lives in at most one dimension per extra group, so these
/// two axes are scored against that separation rather than against the whole
/// residual.
fn group_plane(points: &[(Vec<f32>, Series)]) -> Result<(Basis, bool), String> {
    let dim = points.first().map(|(row, _)| row.len()).unwrap_or(0);
    if dim == 0 || series_in_points(points).len() < 2 {
        return Err("group axes need at least two groups".into());
    }
    let mut global = vec![0.0; dim];
    for (row, _) in points {
        for (slot, value) in global.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    for value in &mut global {
        *value /= points.len() as f64;
    }
    let mut deviations = Vec::new();
    for series in series_in_points(points) {
        let mean = mean_row(points, series);
        let count = points.iter().filter(|(_, item)| *item == series).count() as f64;
        deviations.push(
            mean.iter()
                .zip(&global)
                .map(|(center, overall)| (center - overall) * count.sqrt())
                .collect::<Vec<_>>(),
        );
    }
    let mut basis = basis_from_deviations(&deviations, global)?;
    let y_within = basis.shares[1] < 0.02;
    if y_within {
        basis.axes[1] = within_direction(points, &basis.axes[0])?;
    }
    Ok((basis, y_within))
}

fn within_direction(points: &[(Vec<f32>, Series)], forbidden: &[f64]) -> Result<Vec<f64>, String> {
    let mut rows = Vec::new();
    for series in series_in_points(points) {
        let mean = mean_row(points, series);
        for (row, item) in points {
            if *item != series {
                continue;
            }
            let mut centered = vec![0.0_f32; row.len()];
            let mut along = 0.0;
            for (index, value) in row.iter().enumerate() {
                let delta = f64::from(*value) - mean[index];
                along += delta * forbidden[index];
                centered[index] = delta as f32;
            }
            for (slot, direction) in centered.iter_mut().zip(forbidden) {
                *slot -= (along * direction) as f32;
            }
            rows.push(centered);
        }
    }
    Ok(fit_basis(&rows)?.axes[0].clone())
}

fn shift_plane(shifts: &[(Series, Vec<f32>)]) -> Result<Basis, String> {
    let rows = shifts
        .iter()
        .map(|(_, row)| {
            row.iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dim = rows.first().map(|row| row.len()).unwrap_or(0);
    basis_from_deviations(&rows, vec![0.0; dim])
}

fn basis_from_deviations(rows: &[Vec<f64>], mean: Vec<f64>) -> Result<Basis, String> {
    let count = rows.len();
    if count < 2 || rows[0].is_empty() {
        return Err("a group plane needs at least two directions".into());
    }
    let mut gram = vec![vec![0.0; count]; count];
    let mut total = 0.0;
    for i in 0..count {
        for j in i..count {
            let dot = rows[i]
                .iter()
                .zip(&rows[j])
                .map(|(left, right)| left * right)
                .sum();
            gram[i][j] = dot;
            gram[j][i] = dot;
            if i == j {
                total += dot;
            }
        }
    }
    if total < 1e-12 {
        return Err("the groups have no separation along these residuals".into());
    }
    let (lambda_1, vector_1) = power_axis(&gram, false);
    deflate(&mut gram, lambda_1, &vector_1);
    let (lambda_2, vector_2) = power_axis(&gram, false);
    Ok(Basis {
        axes: [
            combine_axis(rows, &vector_1, lambda_1),
            combine_axis(rows, &vector_2, lambda_2),
        ],
        mean,
        shares: [lambda_1 / total, lambda_2.max(0.0) / total],
    })
}

fn combine_axis(rows: &[Vec<f64>], weights: &[f64], lambda: f64) -> Vec<f64> {
    let mut axis = vec![0.0; rows[0].len()];
    for (row, weight) in rows.iter().zip(weights) {
        for (slot, value) in axis.iter_mut().zip(row) {
            *slot += weight * value;
        }
    }
    scale_axis(axis, lambda)
}

/// Leading eigenvector of a Gram matrix. `center` removes the mean of each
/// iterate, which is the PCA step. The uncentered form keeps a positive shift.
fn power_axis(gram: &[Vec<f64>], center: bool) -> (f64, Vec<f64>) {
    let count = gram.len();
    let mut vector: Vec<f64> = (0..count)
        .map(|index| ((index * 17) % 11) as f64 - 5.0)
        .collect();
    for _ in 0..48 {
        let mut next = matvec(gram, &vector);
        if center && count > 0 {
            let mean = next.iter().sum::<f64>() / count as f64;
            for value in &mut next {
                *value -= mean;
            }
        }
        let norm = l2(&next);
        if norm < 1e-15 {
            break;
        }
        for value in &mut next {
            *value /= norm;
        }
        vector = next;
    }
    let image = matvec(gram, &vector);
    let lambda = vector
        .iter()
        .zip(&image)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .max(0.0);
    (lambda, vector)
}

fn scale_axis(mut axis: Vec<f64>, lambda: f64) -> Vec<f64> {
    let scale = lambda.max(0.0).sqrt().max(1e-12);
    for value in &mut axis {
        *value /= scale;
    }
    let norm = l2(&axis).max(1e-12);
    for value in &mut axis {
        *value /= norm;
    }
    axis
}

fn pca_plane(points: &[(Vec<f32>, Series)]) -> Result<(Vec<(f64, f64, Series)>, [f64; 2]), String> {
    let rows = points
        .iter()
        .map(|(row, _)| row.clone())
        .collect::<Vec<_>>();
    let basis = fit_basis(&rows)?;
    Ok((assign(&project(&rows, &basis), points), basis.shares))
}

fn separating_plane(points: &[(Vec<f32>, Series)]) -> Result<Vec<(f64, f64, Series)>, String> {
    let dim = points[0].0.len();
    let (instruction_mean, instruction_count) = mean_of(points, true, dim);
    let (context_mean, context_count) = mean_of(points, false, dim);
    if instruction_count == 0 || context_count == 0 {
        return Err("both roles are required".into());
    }
    let mut axis = vec![0.0; dim];
    for i in 0..dim {
        axis[i] = instruction_mean[i] - context_mean[i];
    }
    let norm = l2(&axis);
    if norm < 1e-8 {
        return Err("instruction and situation residuals share one centroid".into());
    }
    for value in &mut axis {
        *value /= norm;
    }
    let rows = points
        .iter()
        .map(|(row, _)| {
            let along = dot(row, &axis);
            row.iter()
                .zip(&axis)
                .map(|(value, axis)| value - (along as f32) * (*axis as f32))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let orthogonal = project(&rows, &fit_basis(&rows)?);
    Ok(points
        .iter()
        .zip(orthogonal)
        .map(|((row, role), score)| (dot(row, &axis), score[0], *role))
        .collect())
}

fn mean_of(points: &[(Vec<f32>, Series)], instruction: bool, dim: usize) -> (Vec<f64>, usize) {
    let mut mean = vec![0.0; dim];
    let mut count = 0usize;
    for (row, series) in points {
        if matches!(series, Series::Instruction(_)) != instruction {
            continue;
        }
        count += 1;
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    if count > 0 {
        for value in &mut mean {
            *value /= count as f64;
        }
    }
    (mean, count)
}

/// Top two directions of the centered points, as unit vectors in feature space.
///
/// The Gram matrix is `n` by `n`, which is the small side when each residual
/// is 2,048 wide. A later cloud can be scored in this basis without fitting
/// a new one, so a LoRA shift is not rotated or rescaled out of view.
fn fit_basis(rows: &[Vec<f32>]) -> Result<Basis, String> {
    let n = rows.len();
    let dim = rows.first().map(|row| row.len()).unwrap_or(0);
    if n < 3 || dim == 0 {
        return Err("dimensionality reduction needs at least three residuals".into());
    }
    let mut mean = vec![0.0; dim];
    for row in rows {
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    for value in &mut mean {
        *value /= n as f64;
    }
    let mut gram = vec![vec![0.0; n]; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for k in 0..dim {
                dot += (f64::from(rows[i][k]) - mean[k]) * (f64::from(rows[j][k]) - mean[k]);
            }
            gram[i][j] = dot;
            gram[j][i] = dot;
            if i == j {
                total += dot;
            }
        }
    }
    if total < 1e-12 {
        return Err("residuals have no variance".into());
    }
    let (lambda_1, vector_1) = power_axis(&gram, true);
    deflate(&mut gram, lambda_1, &vector_1);
    let (lambda_2, vector_2) = power_axis(&gram, true);
    Ok(Basis {
        axes: [
            feature_axis(rows, &mean, &vector_1, lambda_1),
            feature_axis(rows, &mean, &vector_2, lambda_2),
        ],
        mean,
        shares: [lambda_1 / total, lambda_2.max(0.0) / total],
    })
}

fn feature_axis(rows: &[Vec<f32>], mean: &[f64], weights: &[f64], lambda: f64) -> Vec<f64> {
    let mut axis = vec![0.0; mean.len()];
    for (row, weight) in rows.iter().zip(weights) {
        for (index, value) in row.iter().enumerate() {
            axis[index] += weight * (f64::from(*value) - mean[index]);
        }
    }
    scale_axis(axis, lambda)
}

fn project(rows: &[Vec<f32>], basis: &Basis) -> Vec<[f64; 2]> {
    rows.iter()
        .map(|row| {
            let mut scores = [0.0, 0.0];
            for (axis_index, axis) in basis.axes.iter().enumerate() {
                let mut dot = 0.0;
                for (value, (center, direction)) in row.iter().zip(basis.mean.iter().zip(axis)) {
                    dot += (f64::from(*value) - center) * direction;
                }
                scores[axis_index] = dot;
            }
            scores
        })
        .collect()
}

fn deflate(gram: &mut [Vec<f64>], lambda: f64, vector: &[f64]) {
    for i in 0..gram.len() {
        for j in 0..gram.len() {
            gram[i][j] -= lambda * vector[i] * vector[j];
        }
    }
}

fn matvec(gram: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    gram.iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn rms_radius(rows: &[Vec<f32>]) -> f64 {
    let dim = rows[0].len();
    let mut mean = vec![0.0; dim];
    for row in rows {
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f64;
    }
    let mut total = 0.0;
    for row in rows {
        for (value, center) in row.iter().zip(&mean) {
            let delta = f64::from(*value) - center;
            total += delta * delta;
        }
    }
    (total / rows.len() as f64).sqrt()
}

fn centroid_distance(left: &[Vec<f32>], right: &[Vec<f32>]) -> f64 {
    let dim = left[0].len();
    let mut delta = vec![0.0; dim];
    for row in left {
        for (slot, value) in delta.iter_mut().zip(row) {
            *slot += f64::from(*value);
        }
    }
    for value in &mut delta {
        *value /= left.len() as f64;
    }
    for row in right {
        for (slot, value) in delta.iter_mut().zip(row) {
            *slot -= f64::from(*value) / right.len() as f64;
        }
    }
    l2(&delta)
}

fn dot(row: &[f32], axis: &[f64]) -> f64 {
    row.iter()
        .zip(axis)
        .map(|(value, axis)| f64::from(*value) * axis)
        .sum()
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn host_f32<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .cast(burn::tensor::FloatDType::F32)
        .into_data()
        .as_slice::<f32>()
        .expect("f32 view of a float tensor")
        .to_vec()
}

fn render_svg(projections: &[Projection]) -> String {
    let panel_w = 460.0;
    let panel_h = 360.0;
    let columns = 2;
    let width = 48.0 + columns as f64 * panel_w;
    let series = present_series(projections);
    let (legend_svg, legend_height) = legend(&series, width - 48.0);
    let mut current_col = 0;
    let mut current_row = 0;
    let mut panel_positions = Vec::new();
    for projection in projections {
        let cols = projection.span_cols.clamp(1, columns);
        if current_col + cols > columns {
            current_col = 0;
            current_row += 1;
        }
        let left = 36.0 + current_col as f64 * panel_w;
        let top = 16.0 + legend_height + current_row as f64 * panel_h;
        let w = cols as f64 * panel_w - 36.0;
        let h = panel_h - 48.0;
        panel_positions.push((left, top, w, h));
        current_col += cols;
        if current_col >= columns {
            current_col = 0;
            current_row += 1;
        }
    }
    let total_rows = if current_col > 0 {
        current_row + 1
    } else {
        current_row
    };
    let height = 28.0 + legend_height + total_rows as f64 * panel_h;
    let mut body = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\">\n"
    );
    body.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#f7f6f3\"/>\n");
    body.push_str(
        "<g font-family=\"ui-sans-serif, sans-serif\" font-size=\"12\" fill=\"#1c1b19\">\n",
    );
    body.push_str(&legend_svg);
    for (projection, (left, top, w, h)) in projections.iter().zip(panel_positions) {
        body.push_str(&panel_svg(projection, left, top, w, h));
    }
    body.push_str("</g></svg>\n");
    body
}

fn panel_svg(projection: &Projection, left: f64, top: f64, width: f64, height: f64) -> String {
    let (x0_lim, x1_lim, y0_lim, y1_lim) = projection
        .frame
        .unwrap_or_else(|| bounds(&projection.coords));
    let mut plot = format!(
        "<rect x=\"{left}\" y=\"{top}\" width=\"{width}\" height=\"{height}\" fill=\"#fffefb\" stroke=\"#d7d3cc\"/>\n"
    );
    if projection.baseline {
        if y0_lim < 0.0 && y1_lim > 0.0 {
            let py = top + (1.0 - (0.0 - y0_lim) / (y1_lim - y0_lim)) * height;
            plot.push_str(&format!(
                "<line x1=\"{left}\" y1=\"{py:.1}\" x2=\"{}\" y2=\"{py:.1}\" stroke=\"#b7b2a9\" stroke-width=\"1\" stroke-dasharray=\"4 3\"/>\n",
                left + width
            ));
        }
        if x0_lim < 0.0 && x1_lim > 0.0 {
            let px = left + (0.0 - x0_lim) / (x1_lim - x0_lim) * width;
            plot.push_str(&format!(
                "<line x1=\"{px:.1}\" y1=\"{top}\" x2=\"{px:.1}\" y2=\"{}\" stroke=\"#b7b2a9\" stroke-width=\"1\" stroke-dasharray=\"4 3\"/>\n",
                top + height
            ));
        }
    }
    for (x, y, role) in &projection.coords {
        let px = left + (*x - x0_lim) / (x1_lim - x0_lim) * width;
        let py = top + (1.0 - (y - y0_lim) / (y1_lim - y0_lim)) * height;
        plot.push_str(&marker(px, py, *role));
    }
    for (x0, y0, x1, y1, series) in &projection.arrows {
        let px0 = left + (*x0 - x0_lim) / (x1_lim - x0_lim) * width;
        let py0 = top + (1.0 - (*y0 - y0_lim) / (y1_lim - y0_lim)) * height;
        let px1 = left + (*x1 - x0_lim) / (x1_lim - x0_lim) * width;
        let py1 = top + (1.0 - (*y1 - y0_lim) / (y1_lim - y0_lim)) * height;
        let color = match series {
            Series::Instruction(kind) => kind_fill(*kind),
            Series::Context => "#b85c38",
        };
        plot.push_str(&format!(
            "<line x1=\"{px0:.1}\" y1=\"{py0:.1}\" x2=\"{px1:.1}\" y2=\"{py1:.1}\" stroke=\"#fffefb\" stroke-width=\"4.4\"/>\n"
        ));
        plot.push_str(&format!(
            "<line x1=\"{px0:.1}\" y1=\"{py0:.1}\" x2=\"{px1:.1}\" y2=\"{py1:.1}\" stroke=\"{color}\" stroke-width=\"2.2\"/>\n"
        ));
        plot.push_str(&arrow_head(px0, py0, px1, py1, color));
    }
    let y_label_y = top + height + 18.0;
    plot.push_str(&format!(
        "<text x=\"{left}\" y=\"{y_label_y}\">{}</text>\n",
        escape(&projection.x_label)
    ));
    plot.push_str(&format!(
        "<text x=\"{left}\" y=\"{}\" fill=\"#5c5852\">{}</text>\n",
        y_label_y + 16.0,
        escape(&projection.y_label)
    ));
    plot
}

fn arrow_head(x0: f64, y0: f64, x1: f64, y1: f64, color: &str) -> String {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let ux = dx / len;
    let uy = dy / len;
    let size = 11.0;
    let base_x = x1 - ux * size;
    let base_y = y1 - uy * size;
    let side_x = -uy * size * 0.42;
    let side_y = ux * size * 0.42;
    format!(
        "<polygon points=\"{x1:.1},{y1:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{color}\"/>\n",
        base_x + side_x,
        base_y + side_y,
        base_x - side_x,
        base_y - side_y
    )
}

fn marker(x: f64, y: f64, series: Series) -> String {
    match series {
        Series::Instruction(kind) => format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3.2\" fill=\"{}\"/>\n",
            kind_fill(kind)
        ),
        Series::Context => format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"5.4\" height=\"5.4\" fill=\"none\" stroke=\"#b85c38\" stroke-width=\"1.4\"/>\n",
            x - 2.7,
            y - 2.7
        ),
    }
}

fn kind_fill(kind: SeedKind) -> &'static str {
    match kind {
        SeedKind::Scene => "#1f4e79",
        SeedKind::Dialogue => "#2f6f4e",
        SeedKind::Monologue => "#6b3fa0",
        SeedKind::CharacterBackground => "#8a5a00",
        SeedKind::GroupConversation => "#8c3a4b",
    }
}

fn series_label(series: Series) -> String {
    match series {
        Series::Instruction(kind) => kind.label().to_string(),
        Series::Context => "situation".to_string(),
    }
}

fn present_series(projections: &[Projection]) -> Vec<Series> {
    let mut found = Vec::new();
    for kind in SeedKind::ALL {
        let series = Series::Instruction(kind);
        if projections
            .iter()
            .any(|projection| projection.coords.iter().any(|(_, _, item)| *item == series))
        {
            found.push(series);
        }
    }
    if projections.iter().any(|projection| {
        projection
            .coords
            .iter()
            .any(|(_, _, item)| *item == Series::Context)
    }) {
        found.push(Series::Context);
    }
    found
}

fn legend(series: &[Series], width: f64) -> (String, f64) {
    let mut svg = String::new();
    let mut x = 36.0;
    let mut y = 20.0;
    for item in series {
        let label = series_label(*item);
        let item_width = 22.0 + label.len() as f64 * 6.6;
        if x > 36.0 && x + item_width > 36.0 + width {
            x = 36.0;
            y += 18.0;
        }
        svg.push_str(&marker(x, y, *item));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\">{}</text>\n",
            x + 10.0,
            y + 4.0,
            escape(&label)
        ));
        x += item_width;
    }
    (svg, y + 12.0)
}

fn bounds(coords: &[(f64, f64, Series)]) -> (f64, f64, f64, f64) {
    let mut x0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y0 = f64::INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for (x, y, _) in coords {
        x0 = x0.min(*x);
        x1 = x1.max(*x);
        y0 = y0.min(*y);
        y1 = y1.max(*y);
    }
    if (x1 - x0).abs() < 1e-9 {
        x0 -= 1.0;
        x1 += 1.0;
    }
    if (y1 - y0).abs() < 1e-9 {
        y0 -= 1.0;
        y1 += 1.0;
    }
    let pad_x = (x1 - x0) * 0.08;
    let pad_y = (y1 - y0) * 0.08;
    (x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y)
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_spans_cover_the_instruction_and_not_the_situation() {
        let situation = "The oven timer rings once.";
        let prompt = seeds::seed(0, SeedKind::Scene, situation).prompt;
        let spans = byte_spans(&prompt, SeedKind::Scene, situation).unwrap();
        let instruction = &prompt[spans.instruction[0].0..spans.instruction[0].1];
        assert!(instruction.contains("Write a scene"));
        assert_eq!(&prompt[spans.context.0..spans.context.1], situation);
        assert_eq!(
            role_at(spans.instruction[0], &spans),
            Some(Role::Instruction)
        );
        assert_eq!(role_at(spans.context, &spans), Some(Role::Context));
        assert_eq!(role_at((0, 4), &spans), None);
    }

    #[test]
    fn a_tighter_instruction_cloud_reports_tightness_above_one() {
        let cloud = Cloud {
            instruction: vec![
                (SeedKind::Scene, vec![0.0, 0.0]),
                (SeedKind::Scene, vec![0.1, 0.0]),
                (SeedKind::Scene, vec![-0.1, 0.0]),
            ],
            context: vec![vec![4.0, 3.0], vec![4.0, -3.0], vec![8.0, 0.0]],
        };
        let report = report("synthetic", &cloud).unwrap();
        assert!(report.tightness > 1.0, "{}", report.tightness);
        assert!(report.centroid_gap > 1.0);
        let points = pooled(&cloud);
        let (coords, shares) = pca_plane(&points).unwrap();
        assert_eq!(coords.len(), 6);
        assert!(shares[0] > shares[1]);
        assert!(shares[0] > 0.5, "pc1 share {}", shares[0]);
    }

    #[test]
    fn an_aggregate_cloud_keeps_every_format_on_one_plane() {
        let cloud = Cloud {
            instruction: SeedKind::ALL
                .into_iter()
                .enumerate()
                .map(|(index, kind)| (kind, vec![index as f32, 0.0]))
                .collect(),
            context: vec![vec![0.0, 5.0], vec![1.0, 6.0], vec![2.0, 4.0]],
        };
        let panels = panels_for("all formats", &cloud).unwrap();
        let kinds = panels[0]
            .coords
            .iter()
            .filter_map(|(_, _, series)| match series {
                Series::Instruction(kind) => Some(*kind),
                Series::Context => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(kinds.len(), SeedKind::ALL.len());
        assert!(
            panels[0]
                .coords
                .iter()
                .any(|(_, _, series)| *series == Series::Context)
        );
    }

    #[test]
    fn a_lora_shift_stays_on_one_scale_and_in_the_change_panel() {
        let base = Cloud {
            instruction: vec![
                (SeedKind::Scene, vec![0.0, 0.0]),
                (SeedKind::Scene, vec![0.2, 0.0]),
                (SeedKind::Scene, vec![-0.2, 0.0]),
            ],
            context: vec![vec![6.0, 1.0], vec![6.0, -1.0], vec![7.0, 0.0]],
        };
        let adapted = Cloud {
            instruction: vec![
                (SeedKind::Scene, vec![0.0, 3.0]),
                (SeedKind::Scene, vec![0.2, 3.0]),
                (SeedKind::Scene, vec![-0.2, 3.0]),
            ],
            context: base.context.clone(),
        };
        let (instruction, situation) = (
            relative_movement(&instruction_rows(&base), &instruction_rows(&adapted)).unwrap(),
            relative_movement(&base.context, &adapted.context).unwrap(),
        );
        assert!(instruction > 1.0, "{instruction}");
        assert!(situation < 1e-6, "{situation}");
        let panels = paired_panels(&base, &adapted).unwrap();
        assert_eq!(panels.len(), 7);
        let scene = Series::Instruction(SeedKind::Scene);
        let (shift_x, shift_y) = centroid_of(&panels[0].coords, scene).unwrap();
        let (context_x, context_y) = centroid_of(&panels[0].coords, Series::Context).unwrap();
        assert!((shift_x - context_x).hypot(shift_y - context_y) > 0.5);
        assert!(!panels[0].arrows.is_empty());
        assert_eq!(panels[0].span_cols, 2);
        let (after_a_scene_x, after_a_scene_y) = centroid_of(&panels[2].coords, scene).unwrap();
        let (after_a_context_x, after_a_context_y) =
            centroid_of(&panels[2].coords, Series::Context).unwrap();
        let separated =
            (after_a_scene_x - after_a_context_x).hypot(after_a_scene_y - after_a_context_y);
        assert!(separated > 0.05, "{separated}");
        assert!(panels[2].x_label.contains("Option A"));
    }

    #[test]
    fn a_shift_off_the_group_plane_is_drawn_on_the_scatter() {
        let base = Cloud {
            instruction: vec![
                (SeedKind::Scene, vec![0.0, 0.4, 0.0]),
                (SeedKind::Scene, vec![0.2, -0.4, 0.0]),
                (SeedKind::Scene, vec![-0.2, 0.0, 0.0]),
            ],
            context: vec![
                vec![6.0, 0.3, 0.0],
                vec![6.2, -0.3, 0.0],
                vec![5.8, 0.0, 0.0],
            ],
        };
        let adapted = Cloud {
            instruction: base
                .instruction
                .iter()
                .map(|(kind, row)| (*kind, vec![row[0], row[1], row[2] + 0.05]))
                .collect(),
            context: base
                .context
                .iter()
                .map(|row| vec![row[0], row[1], row[2] + 0.01])
                .collect(),
        };
        let panels = paired_panels(&base, &adapted).unwrap();
        assert_eq!(panels.len(), 7);
        let scene = Series::Instruction(SeedKind::Scene);
        let (scene_x, scene_y) = centroid_of(&panels[2].coords, scene).unwrap();
        let (context_x, context_y) = centroid_of(&panels[2].coords, Series::Context).unwrap();
        let separated = (scene_x - context_x).hypot(scene_y - context_y);
        assert!(separated > 0.05, "{separated}");
        assert!(panels[2].x_label.contains("Option A"));
    }
}
