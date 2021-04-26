use std::fmt;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Index, IndexMut};
use crate::graph::EdgeType::Undirected;


//TODO: Make reference graph that just stores references to items

#[derive(Eq, PartialEq)]
pub enum EdgeType {
    Directed,
    Undirected
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub struct GraphIndex {
    index: u64
}
impl GraphIndex {
    pub fn from<T: Hash + ?Sized>(x: &T) -> Self {
        let mut d = DefaultHasher::default();
        x.hash(&mut d);
        GraphIndex {
            index: d.finish()
        }
    }
}


#[allow(dead_code)]
pub struct Graph<T> {
    edges: HashMap<GraphIndex, (T, HashSet<GraphIndex>)>
}

impl<T: Eq + Hash> Graph<T> {
    pub fn new() -> Graph<T> {
        Graph {
            edges: HashMap::new()
        }
    }

    //TODO: Make it so you can do the following add_vertex(2).edge(3)
    pub fn add_vertex(&mut self, v: T) -> GraphIndex {
        let g = GraphIndex::from(&v);
        self.edges.entry(g).or_insert((v, HashSet::new()));
        g
    }

    pub fn add_directed_edge(&mut self, v1: GraphIndex, v2: GraphIndex) {
        if !self.edges.contains_key(&v1) || !self.edges.contains_key(&v2) {
            return;
        }
        let (_, edges) = self.edges.get_mut(&v1).unwrap();
        edges.insert(v2);
    }

    pub fn contains(&self, v: &T) -> bool {
        self.edges.contains_key(&GraphIndex::from(v))
    }

    pub fn dfs_iter(&self, v: GraphIndex) -> GraphIter<T> {
        let mut s = VecDeque::new();
        s.push_back(v);
        GraphIter {
            graph: self,
            visited: HashSet::new(),
            data: s,
            is_dfs: true
        }
    }

    // pub fn dfs_iter_mut<'a>(&'a mut self, v: &'a GraphIndex) -> GraphIterMut<'a, T> {
    //     let mut s = VecDeque::new();
    //     s.push_back(v);
    //     GraphIterMut {
    //         graph: self,
    //         visited: HashSet::new(),
    //         data: s,
    //         is_dfs: true
    //     }
    // }

    pub fn bfs_iter(&self, v: GraphIndex) -> GraphIter<T> {
        let mut s = VecDeque::new();
        s.push_back(v);
        GraphIter {
            graph: self,
            visited: HashSet::new(),
            data: s,
            is_dfs: false
        }
    }

    // pub fn bfs_iter_mut(&'amut self, v: GraphIndex) -> GraphIterMut<T> {
    //     let mut s = VecDeque::new();
    //     s.push_back(v);
    //     GraphIterMut {
    //         graph: self,
    //         visited: HashSet::new(),
    //         data: s,
    //         is_dfs: false
    //     }
    // }
}

impl<T> Index<GraphIndex> for Graph<T> {
    type Output = T;

    fn index(&self, index: GraphIndex) -> &Self::Output {
        match self.edges.get(&index) {
            None => {
                panic!("No Vertex in Graph")
            },
            Some((v, _)) => v
        }
    }
}

impl<T> IndexMut<GraphIndex> for Graph<T> {

    fn index_mut(&mut self, index: GraphIndex) -> &mut Self::Output {
        match self.edges.get_mut(&index) {
            None => {
                panic!("No Vertex in Graph")
            },
            Some((v, _)) => v
        }
    }
}

//Assumes Directed edges from T to T
impl<T: Eq + Hash> From<(Vec<(T, T)>, EdgeType)> for Graph<T> {
    fn from((edges, e_type): (Vec<(T, T)>, EdgeType)) -> Self {
        let mut g = Graph::new();
        for (i, j) in edges {
            let i = g.add_vertex(i);
            let j = g.add_vertex(j);
            g.add_directed_edge(i, j);
            if e_type == Undirected {
                g.add_directed_edge(j, i);
            }
        }
        g
    }
}


impl<T: ToString> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, v) in &self.edges {
            write!(f, "{} => {{", v.0.to_string())?;
            let edges = &v.1;
            for (i, v) in edges.iter().enumerate() {
                let v = self.edges.get(v);
                match v {
                    Some((x, _)) => {
                        if i < edges.len() - 1 {
                            write!(f, "{}, ", x.to_string())?;
                        } else {
                            write!(f, "{}", x.to_string())?;
                        }
                    }
                    _ => {}
                }
            }
            writeln!(f, "}}")?;
        }
        Ok(())
    }
}

pub struct GraphIter<'a, T> {
    graph: &'a Graph<T>,
    visited: HashSet<GraphIndex>,
    data: VecDeque<GraphIndex>,
    is_dfs: bool
}

impl<'a, T: Eq + Hash> Iterator for GraphIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }
        let v;
        if self.is_dfs {
            v = self.data.pop_back().unwrap();
        } else {
            v = self.data.pop_front().unwrap();
        }
        return match self.graph.edges.get(&v) {
            None => None,
            Some((t, edges)) => {
                for k in edges {
                    if !self.visited.contains(k) {
                        self.data.push_back(*k);
                    }
                }
                self.visited.insert(v);
                Some(t)
            }
        };
    }
}

// pub struct GraphIterMut<'a, T: 'a> {
//     graph: &'a mut Graph<T>,
//     visited: HashSet<&'a GraphIndex>,
//     data: VecDeque<&'a GraphIndex>,
//     is_dfs: bool
// }
//
// impl<'a, T: 'a + Eq + Hash> Iterator for GraphIterMut<'a, T> {
//     type Item = &'a mut T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.data.is_empty() {
//             return None;
//         }
//         let v;
//         if self.is_dfs {
//             v = self.data.pop_back().unwrap();
//         } else {
//             v = self.data.pop_front().unwrap();
//         }
//         let (i, _) = self.graph.edges.get_mut(v).unwrap();
//         Some(i)
//         // return match k {
//         //     None => None,
//         //     Some((t, edges)) => {
//         //         for k in edges.iter() {
//         //             if !self.visited.contains(k) {
//         //                 self.data.push_back(*k);
//         //             }
//         //         }
//         //         self.visited.insert(v);
//         //         Some(&mut t)
//         //     }
//         // };
//     }
// }