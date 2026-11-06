use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub other: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Character {
    pub id: String,
    pub name: String,
    pub desires: Vec<String>,
    pub beliefs: Vec<String>,
    pub knowledge: Vec<String>,
    pub relationships: Vec<Relationship>,
}
