//! Read the coherence pass the model returns after the beats are written.

#[derive(Debug)]
pub enum QaReading {
    Coherent,
    Revised(Vec<String>),
    Unreadable,
}

pub fn read_qa(text: &str, sections: usize) -> QaReading {
    let trimmed = text.trim();
    if trimmed.is_empty() || sections == 0 {
        return QaReading::Unreadable;
    }
    let first = trimmed.lines().next().unwrap_or("").trim();
    let head = first
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_uppercase();
    if head.starts_with("COHERENT") {
        return QaReading::Coherent;
    }
    let body = if head.starts_with("REVISED") {
        trimmed.lines().skip(1).collect::<Vec<_>>().join("\n")
    } else {
        trimmed.to_string()
    };
    let pieces = split_sections(&body);
    if pieces.len() == sections && pieces.iter().all(|piece| !piece.trim().is_empty()) {
        QaReading::Revised(pieces)
    } else {
        QaReading::Unreadable
    }
}

fn split_sections(body: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current: Option<String> = None;
    for line in body.lines() {
        if line.trim().to_ascii_lowercase().starts_with("title:") {
            if let Some(prose) = current.take() {
                sections.push(strip_prose_label(&prose));
            }
            current = Some(String::new());
            continue;
        }
        if let Some(buf) = current.as_mut() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
    }
    if let Some(prose) = current {
        sections.push(strip_prose_label(&prose));
    }
    sections
}

fn strip_prose_label(text: &str) -> String {
    let trimmed = text.trim();
    let rest = trimmed
        .strip_prefix("prose:")
        .or_else(|| trimmed.strip_prefix("Prose:"))
        .unwrap_or(trimmed);
    rest.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_is_left_alone() {
        assert!(matches!(read_qa("COHERENT", 3), QaReading::Coherent));
    }

    #[test]
    fn revised_sections_keep_order() {
        let text = "\
REVISED
title: The door
prose:
Mara waited.
title: The answer
prose:
Ellis told her the truth.
";
        match read_qa(text, 2) {
            QaReading::Revised(pieces) => {
                assert_eq!(pieces, vec!["Mara waited.", "Ellis told her the truth."]);
            }
            other => panic!("expected a revision, got a different reading: {other:?}"),
        }
    }

    #[test]
    fn a_short_revision_is_rejected() {
        let text = "\
REVISED
title: Only one
prose:
Not enough sections.
";
        assert!(matches!(read_qa(text, 2), QaReading::Unreadable));
    }
}
