use pyo3::prelude::*;
use std::collections::HashMap;
use std::cmp::Ordering;

const INITIAL_BOARD: [[i8; 9]; 9] = [
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 2, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
];

struct Board {
    board: Vec<Vec<i8>>,
    past_legal_moves: Vec<HashMap<String, Vec<(i8, i8)>>>,
    past_moves: Vec<((i8, i8), (i8, i8))>,
    current_player: i8,
    legal_moves: HashMap<String, Vec<(i8, i8)>>,
}

impl Board {
    fn convert_coord_to_str(pos: (i8, i8)) -> String {
        // Pos in format (8, 0) to A1
        let col = (b'A' + (pos.1 as u8)) as char;
        let row = 9 - pos.0;
        format!("{}{}", col, row)
    }

    fn convert_str_to_coord(pos: &str) -> (i8, i8) {
        // Pos in format A1 to (8, 0)
        let col = (pos.chars().nth(0).unwrap() as i8) - 65;
        let row = 9 - (pos.chars().nth(1).unwrap().to_digit(10).unwrap() as i8);
        (row, col)
    }

    fn is_game_over(&self) -> i8 {
        if self.board[0].contains(&1) {
            return 1;
        }
        if self.board[8].contains(&2) {
            return 2;
        }
        return 0;
    }

    fn get_all_possible_moves(&self, player: i8) -> HashMap<String, Vec<(i8, i8)>> {
        let mut legal_moves: HashMap<String, Vec<(i8, i8)>> = HashMap::new();
        let mut capturers: Vec<(i8, i8)> = Vec::new();

        for i in 0..9 {
            for j in 0..9 {
                if self.board[i][j] == player {
                    let moves: Vec<(i8, i8)> = self.get_possible_moves((i, j));
                    let key = format!("{}{}", i, j);
                    legal_moves.insert(key.clone(), moves.clone());

                    let mut capture_move = vec![];
                    for &move_ in &moves {
                        if ((i as isize) - (move_.0 as isize)).abs() > 1 {
                            capture_move.push(move_);
                        }
                    }

                    if !capture_move.is_empty() {
                        capturers.push((i as i8, j as i8));
                        legal_moves.insert(key, capture_move);
                    }
                }
            }
        }

        if !capturers.is_empty() {
            let mut only_captures: HashMap<String, Vec<(i8, i8)>> = HashMap::new();
            for &capturer in &capturers {
                let key = format!("{}{}", capturer.0, capturer.1);
                if let Some(moves) = legal_moves.get(&key) {
                    only_captures.insert(key, moves.clone());
                }
            }
            return only_captures;
        }

        legal_moves
    }

    fn get_possible_moves(&self, piece: (usize, usize)) -> Vec<(i8, i8)> {
        let player: i8 = self.board[piece.0][piece.1];

        if player == 0 {
            return vec![];
        }

        let mut moves: Vec<(i8, i8)> = vec![
            (piece.0 as i8, (piece.1 as i8) + 1),
            (piece.0 as i8, (piece.1 as i8) - 1)
        ];

        if player == 1 {
            moves.push(((piece.0 as i8) - 1, piece.1 as i8));
            if piece.0 > 1 && piece.1 < 7 {
                if self.board[piece.0 - 1][piece.1 + 1] == 2 {
                    moves.push(((piece.0 as i8) - 2, (piece.1 as i8) + 2));
                }
            }
            if piece.0 > 1 && piece.1 > 1 {
                if self.board[piece.0 - 1][piece.1 - 1] == 2 {
                    moves.push(((piece.0 as i8) - 2, (piece.1 as i8) - 2));
                }
            }
        } else {
            moves.push(((piece.0 as i8) + 1, piece.1 as i8));
            if piece.0 < 7 && piece.1 < 7 {
                if self.board[piece.0 + 1][piece.1 + 1] == 1 {
                    moves.push(((piece.0 as i8) + 2, (piece.1 as i8) + 2));
                }
            }
            if piece.0 < 7 && piece.1 > 1 {
                if self.board[piece.0 + 1][piece.1 - 1] == 1 {
                    moves.push(((piece.0 as i8) + 2, (piece.1 as i8) - 2));
                }
            }
        }

        return moves
            .into_iter()
            .filter(
                |&(x, y)|
                    x < 9 && y < 9 && x >= 0 && y >= 0 && self.board[x as usize][y as usize] == 0
            )
            .collect();
    }

    fn calculate_legal_moves(&mut self) {
        if self.is_game_over() > 0 {
            self.legal_moves = HashMap::new();
        }
        self.legal_moves = self.get_all_possible_moves(self.current_player);
    }

    fn make_move(&mut self, piece_: (i8, i8), move_: (i8, i8)) {
        let piece = (piece_.0 as usize, piece_.1 as usize);
        let move_ = (move_.0 as usize, move_.1 as usize);

        self.board[move_.0][move_.1] = self.board[piece.0][piece.1];
        self.board[piece.0][piece.1] = 0;

        if ((piece.0 as isize) - (move_.0 as isize)).abs() > 1 {
            let captured_piece = (
                ((piece.0 as isize) + (move_.0 as isize)) / 2,
                ((piece.1 as isize) + (move_.1 as isize)) / 2,
            );
            self.board[captured_piece.0 as usize][captured_piece.1 as usize] = 0;
        }

        self.past_moves.push((piece_, (move_.0 as i8, move_.1 as i8)));
        self.past_legal_moves.push(self.legal_moves.clone());
        self.current_player = if self.current_player == 1 { 2 } else { 1 };

        self.calculate_legal_moves();
    }

    fn undo_move(&mut self) {
        if self.past_moves.is_empty() {
            return;
        }

        let (start, end) = self.past_moves.pop().unwrap();
        if ((start.0 as isize) - (end.0 as isize)).abs() > 1 {
            let captured_piece = (
                ((start.0 as isize) + (end.0 as isize)) / 2,
                ((start.1 as isize) + (end.1 as isize)) / 2,
            );
            self.board[captured_piece.0 as usize][captured_piece.1 as usize] = self.current_player;
        }

        self.board[start.0 as usize][start.1 as usize] = self.board[end.0 as usize][end.1 as usize];
        self.board[end.0 as usize][end.1 as usize] = 0;
        self.current_player = if self.current_player == 1 { 2 } else { 1 };
        self.legal_moves = self.past_legal_moves.pop().unwrap();
    }
}

struct Engine {
    player: i8,
    depth: i32,
    transposition_table: HashMap<String, (f64, Vec<String>)>,
    table_hits: i32,
}

impl Engine {
    fn evaluate(&self, board: &Board) -> f64 {
        let player_prospective =
            (if board.current_player == self.player { 1 } else { -1 }) *
            (if self.player == 1 { 1 } else { -1 });

        let ones_count: i32 = board.board
            .iter()
            .map(
                |row|
                    row
                        .iter()
                        .filter(|&&x| x == 1)
                        .count() as i32
            )
            .sum();
        let twos_count: i32 = board.board
            .iter()
            .map(
                |row|
                    row
                        .iter()
                        .filter(|&&x| x == 2)
                        .count() as i32
            )
            .sum();

        let material_diff = ones_count - twos_count;

        match board.is_game_over() {
            1 => {
                return 1000.0 * (player_prospective as f64);
            }
            2 => {
                return -1000.0 * (player_prospective as f64);
            }
            _ => {}
        }

        let mut ones_rows: Vec<usize> = board.board
            .iter()
            .enumerate()
            .flat_map(|(row_index, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(_, &value)| value == 1)
                    .map(move |(_col_index, _)| row_index)
            })
            .collect();

        ones_rows = ones_rows
            .iter()
            .map(|&x| 9 - x)
            .collect();

        let mut twos_rows: Vec<usize> = board.board
            .iter()
            .enumerate()
            .flat_map(|(row_index, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(_, &value)| value == 2)
                    .map(move |(_col_index, _)| row_index)
            })
            .collect();

        twos_rows = twos_rows
            .iter()
            .rev()
            .map(|&x| x + 1)
            .collect();

        let avg_ones = weighted_average(&ones_rows);
        let avg_twos = weighted_average(&twos_rows);

        // println!("Ones rows: {:?}", ones_rows);
        // println!("Twos rows: {:?}", twos_rows);
        // println!("Avg ones: {}", avg_ones);
        // println!("Avg twos: {}", avg_twos);

        let avg_diff: f64 = (avg_ones - avg_twos) as f64;

        return ((material_diff as f64) * 5.0 + avg_diff * 10.0) * (player_prospective as f64);
    }

    fn get_best_move(&mut self, board: &mut Board) -> (String, f64) {
        let mut best_eval = f64::NEG_INFINITY;
        let mut best_move: ((i8, i8), (i8, i8)) = ((0, 0), (0, 0));
        let mut best_move_sequence = Vec::new();
        let mut alpha = f64::NEG_INFINITY;
        let beta = f64::INFINITY;

        let moves = self.get_ordered_moves(board); // Use ordered moves
        if board.board == INITIAL_BOARD {
            return ("D4-D5".to_string(), 0.0);
        }

        if board.legal_moves.len() == 1 {
            return (
                format!(
                    "{}-{}",
                    Board::convert_coord_to_str(moves[0].0),
                    Board::convert_coord_to_str(moves[0].1)
                ),
                self.evaluate(board) *
                    (if self.player == board.current_player { 1.0 } else { -1.0 }),
            );
        }

        for (piece_coords, move_) in moves {
            board.make_move(piece_coords, move_);
            // if board.is_game_over() == self.player {
            //     return (
            //         format!(
            //             "{}-{}",
            //             Board::convert_coord_to_str(piece_coords),
            //             Board::convert_coord_to_str(move_)
            //         ),
            //         1000.0 * (if self.player == board.current_player { 1.0 } else { -1.0 }),
            //     );
            // }
            let (mut eval, move_sequence) = self.negamax(
                board,
                self.depth - 1,
                -beta,
                -alpha,
                true
            );
            eval = -eval;

            if eval > best_eval {
                best_eval = eval;
                best_move = (piece_coords, move_);
                best_move_sequence = vec![
                    format!(
                        "{}-{}",
                        Board::convert_coord_to_str(piece_coords),
                        Board::convert_coord_to_str(move_)
                    )
                ];
                best_move_sequence.extend(move_sequence);
            }
            board.undo_move();

            alpha = alpha.max(eval);
            if alpha >= beta {
                break;
            }
        }

        println!("Positions evaluated: {}", self.transposition_table.len());
        println!("Table hits: {}", self.table_hits);

        self.transposition_table.clear();

        let best_move_str = format!(
            "{}-{}",
            Board::convert_coord_to_str(best_move.0),
            Board::convert_coord_to_str(best_move.1)
        );

        let mut expected_eval: f64 = 0.0;
        for move_ in best_move_sequence.clone() {
            let coords: Vec<&str> = move_.split('-').collect();
            let start = Board::convert_str_to_coord(coords[0]);
            let end: (i8, i8) = Board::convert_str_to_coord(coords[1]);
            board.make_move(start, end);
            expected_eval =
                self.evaluate(board) *
                (if self.player == board.current_player { 1.0 } else { -1.0 });
            println!("{}: {}", move_, expected_eval);
        }

        return (best_move_str, expected_eval);
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: i32,
        mut alpha: f64,
        beta: f64,
        hash_table: bool
    ) -> (f64, Vec<String>) {
        // Check in hash table
        let board_hash = format!("{:?}", board.board);
        if hash_table {
            if let Some(&(eval, ref move_sequence)) = self.transposition_table.get(&board_hash) {
                self.table_hits += 1;
                return (eval, move_sequence.clone());
            }
        }

        // Check if game is over or reached max depth
        if depth == 0 || board.is_game_over() > 0 {
            let eval = self.evaluate(board);
            self.transposition_table.insert(board_hash, (eval, Vec::new()));
            return (eval, Vec::new());
        }

        let mut best_eval = f64::NEG_INFINITY;
        let mut best_move_sequence = Vec::new();

        let moves = self.get_ordered_moves(board); // Use ordered moves

        for (piece_coords, move_) in moves {
            board.make_move(piece_coords, move_);
            // if (piece_coords.0 - move_.0).abs() > 1 {
            //     depth += 1;
            // }
            let (mut eval, move_sequence) = self.negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                hash_table
            );
            eval = -eval;
            if eval > best_eval {
                best_eval = eval;
                best_move_sequence = vec![
                    format!(
                        "{}-{}",
                        Board::convert_coord_to_str(piece_coords),
                        Board::convert_coord_to_str(move_)
                    )
                ];
                best_move_sequence.extend(move_sequence);
            }
            alpha = alpha.max(eval);
            board.undo_move();
            if alpha >= beta {
                break;
            }
        }

        self.transposition_table.insert(board_hash, (best_eval, best_move_sequence.clone()));
        (best_eval, best_move_sequence)
    }

    fn get_ordered_moves(&self, board: &Board) -> Vec<((i8, i8), (i8, i8))> {
        let mut moves: Vec<((i8, i8), (i8, i8))> = Vec::new();
        for (piece, piece_moves) in &board.legal_moves {
            let piece_coords = (
                piece.chars().nth(0).unwrap().to_digit(10).unwrap() as i8,
                piece.chars().nth(1).unwrap().to_digit(10).unwrap() as i8,
            );
            for &move_ in piece_moves {
                moves.push((piece_coords, move_).clone());
            }
        }

        moves.sort_by(|a: &((i8, i8), (i8, i8)), b: &((i8, i8), (i8, i8))| {
            if a.0 > b.0 {
                return Ordering::Less;
            } else {
                return Ordering::Greater;
            }
        });

        if board.current_player == 1 {
            moves.reverse();
        }

        moves
    }
}

fn weighted_average(values: &Vec<usize>) -> f64 {
    // Define a base for the exponential function
    let base: f64 = 1.05;

    // Calculate the weights using an exponential function
    let weights: Vec<f64> = values
        .iter()
        .map(|&v| base.powf(v as f64))
        .collect();

    // Compute the weighted sum of the values
    let weighted_sum: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| (v as f64) * w)
        .sum();

    // Compute the sum of the weights
    let sum_of_weights: f64 = weights.iter().sum();

    // Calculate the weighted average
    weighted_sum / sum_of_weights
}

#[pyfunction]
fn get_best_move(
    board: Vec<Vec<i8>>,
    board_current_plater: i8,
    player: i8,
    depth: i32
) -> (String, f64) {
    let mut engine = Engine {
        player,
        depth,
        transposition_table: HashMap::new(),
        table_hits: 0,
    };
    let mut board = Board {
        board: board.clone(),
        past_legal_moves: vec![],
        past_moves: vec![],
        current_player: board_current_plater,
        legal_moves: HashMap::new(),
    };
    board.calculate_legal_moves();

    let (best_move, expected_eval) = engine.get_best_move(&mut board);
    if best_move == "A9-A9" {
        board.calculate_legal_moves();
        println!("{:?}", board.legal_moves);
    }

    return (best_move, expected_eval);
}

#[pyfunction]
fn evaluate_stand_alone(board: Vec<Vec<i8>>, board_current_plater: i8, player: i8) -> f64 {
    let engine = Engine {
        player,
        depth: 0,
        transposition_table: HashMap::new(),
        table_hits: 0,
    };
    let mut board = Board {
        board: board.clone(),
        past_legal_moves: vec![],
        past_moves: vec![],
        current_player: board_current_plater,
        legal_moves: HashMap::new(),
    };
    board.calculate_legal_moves();

    return engine.evaluate(&board);
}

#[pymodule]
fn fianco_tournament(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(is_game_over, m)?)?;
    // m.add_function(wrap_pyfunction!(get_all_possible_moves, m)?)?;
    // m.add_function(wrap_pyfunction!(calculate_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_stand_alone, m)?)?;
    Ok(())
}
