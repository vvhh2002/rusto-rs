//! Conservative grouping of OCR fragments. Geometry determines whether two
//! fragments share a row; a nearby x coordinate alone never joins separate rows.
use crate::{Frame, TextResult};

fn valid(frame: &Frame) -> bool {
    [
        frame.left,
        frame.top,
        frame.width,
        frame.height,
        frame.left + frame.width,
        frame.top + frame.height,
    ]
    .iter()
    .all(|v| v.is_finite())
        && frame.width > 0.0
        && frame.height > 0.0
}

fn same_row(a: &Frame, b: &Frame, tolerance: f32) -> bool {
    if !valid(a) || !valid(b) {
        return false;
    }
    let height = a.height.min(b.height);
    if a.height.max(b.height) > 2.0 * height {
        return false;
    }
    let vertical_overlap = ((a.top + a.height).min(b.top + b.height) - a.top.max(b.top)).max(0.0);
    let center_distance = ((a.top + a.height / 2.0) - (b.top + b.height / 2.0)).abs();
    let horizontal_overlap =
        ((a.left + a.width).min(b.left + b.width) - a.left.max(b.left)).max(0.0);
    vertical_overlap >= height * 0.8
        && center_distance <= height * tolerance
        && horizontal_overlap == 0.0
}

/// Stable row groups, each in left-to-right order. Every pair in a row must
/// agree: an intermediate fragment cannot bridge two adjacent text lines.
pub(crate) fn line_groups(entries: &[TextResult], tolerance: f32) -> Vec<Vec<usize>> {
    let tolerance = if tolerance.is_finite() {
        tolerance.max(0.0)
    } else {
        0.5
    };
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| {
        entries[a]
            .frame
            .top
            .total_cmp(&entries[b].frame.top)
            .then(entries[a].frame.left.total_cmp(&entries[b].frame.left))
    });
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let candidate = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                g.iter()
                    .all(|&j| same_row(&entries[i].frame, &entries[j].frame, tolerance))
            })
            .min_by(|(_, a), (_, b)| {
                let center = |j: usize| entries[j].frame.top + entries[j].frame.height / 2.0;
                (center(i) - center(a[0]))
                    .abs()
                    .total_cmp(&(center(i) - center(b[0])).abs())
            })
            .map(|(index, _)| index);
        if let Some(index) = candidate {
            groups[index].push(i);
        } else {
            groups.push(vec![i]);
        }
    }
    for group in &mut groups {
        group.sort_by(|&a, &b| entries[a].frame.left.total_cmp(&entries[b].frame.left));
    }
    groups
}

pub(crate) fn merge_line_fragments(entries: &[TextResult], y: f32, x: f32) -> Vec<TextResult> {
    let gap_factor = if x.is_finite() { x.max(0.0) } else { 0.4 };
    let mut result = Vec::new();
    for group in line_groups(entries, y) {
        let mut fragment: Option<TextResult> = None;
        let mut previous: Option<&TextResult> = None;
        for i in group {
            let entry = &entries[i];
            let can_merge = previous.is_some_and(|prev| {
                let gap = entry.frame.left - (prev.frame.left + prev.frame.width);
                same_row(&prev.frame, &entry.frame, y)
                    && gap <= prev.frame.height.min(entry.frame.height) * gap_factor
            });
            if can_merge {
                if let Some(current) = &mut fragment {
                    current.text.push(' ');
                    current.text.push_str(&entry.text);
                    current.score = current.score.min(entry.score);
                    let left = current.frame.left.min(entry.frame.left);
                    let top = current.frame.top.min(entry.frame.top);
                    let right = (current.frame.left + current.frame.width)
                        .max(entry.frame.left + entry.frame.width);
                    let bottom = (current.frame.top + current.frame.height)
                        .max(entry.frame.top + entry.frame.height);
                    current.box_points =
                        [(left, top), (right, top), (right, bottom), (left, bottom)];
                    current.frame = Frame::from_points(&current.box_points);
                }
            } else {
                if let Some(current) = fragment.take() {
                    result.push(current);
                }
                fragment = Some(entry.clone());
            }
            previous = Some(entry);
        }
        if let Some(current) = fragment {
            result.push(current);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(s: &str, x: f32, y: f32, w: f32, h: f32) -> TextResult {
        let points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
        TextResult {
            text: s.into(),
            score: 0.9,
            frame: Frame::from_points(&points),
            box_points: points,
        }
    }
    fn strings(input: &[TextResult]) -> Vec<String> {
        merge_line_fragments(input, 0.5, 0.4)
            .into_iter()
            .map(|r| r.text)
            .collect()
    }
    #[test]
    fn dense_small_print_is_not_merged_using_large_page_text_height() {
        let input = vec![
            text("Heading", 0., 0., 120., 36.),
            text("Other", 0., 60., 120., 36.),
            text("Large", 0., 120., 120., 36.),
            text("first", 15., 180., 120., 8.),
            text("second", 10., 189., 120., 8.),
        ];
        assert_eq!(
            strings(&input),
            ["Heading", "Other", "Large", "first", "second"]
        );
    }
    #[test]
    fn joins_nearby_fragments_but_preserves_columns_and_unrelated_overlaps() {
        let input = vec![
            text("Hello", 0., 0., 40., 10.),
            text("world", 43., 1., 40., 10.),
            text("column", 160., 0., 40., 10.),
        ];
        assert_eq!(strings(&input), ["Hello world", "column"]);
        let input = vec![
            text("upper", 10., 0., 100., 20.),
            text("lower", 0., 8., 100., 20.),
        ];
        assert_eq!(strings(&input), ["upper", "lower"]);
    }
    #[test]
    fn intermediate_box_cannot_bridge_distinct_rows() {
        let input = vec![
            text("one", 0., 0., 10., 10.),
            text("two", 13., 2., 10., 10.),
            text("three", 26., 4., 10., 10.),
        ];
        assert_eq!(strings(&input), ["one two", "three"]);
    }
    #[test]
    fn preserves_multilingual_content_and_deterministic_order() {
        let input = vec![
            text("문서", 100., 40., 20., 10.),
            text("日本語", 10., 20., 30., 10.),
            text("Français", 10., 0., 70., 10.),
        ];
        assert_eq!(strings(&input), ["Français", "日本語", "문서"]);
        let a = merge_line_fragments(&input, 0.5, 0.4);
        assert_eq!(a[0].box_points, input[2].box_points);
    }
    #[test]
    fn overlapping_column_headers_and_offset_paragraphs_are_not_joined() {
        let input = vec![
            text("Account", 400., 560., 145., 17.),
            text("Volume", 542., 559., 57., 19.),
        ];
        assert_eq!(merge_line_fragments(&input, 0.5, 0.4).len(), 2);
        let input = vec![
            text("left paragraph", 0., 100., 600., 30.),
            text("right paragraph", 606., 84., 600., 37.),
        ];
        assert_eq!(merge_line_fragments(&input, 0.5, 0.4).len(), 2);
    }
    #[test]
    fn invalid_or_empty_geometry_never_drops_text_or_panics() {
        let mut input = vec![
            text("flat", 0., 0., 10., 0.),
            text("valid", 0., 1., 10., 10.),
        ];
        input.push(text("invalid", f32::NAN, 1., 10., 10.));
        assert_eq!(merge_line_fragments(&input, 0.5, 0.4).len(), 3);
        assert!(merge_line_fragments(&[], 0.5, 0.4).is_empty());
    }
}
