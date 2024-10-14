set -e
maturin build --release --target x86_64-pc-windows-gnu -i python3.10

cp /Users/marcon21/Documents/UM/fianco-tournament/target/x86_64-pc-windows-gnu/release/fianco_tournament.dll /Users/marcon21/Documents/UM/fianco-tournament/fianco_tournament_win.pyd