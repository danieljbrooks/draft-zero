//! `dzk play ...`, `dzk summarize ...` and `dzk bench ...` (see README.md).

use dzk::nntools::{run_nnbench, run_parity, NnBenchArgs};
use dzk::record::GameLimits;
use dzk::runner::{run_bench, run_play, run_summarize, BenchArgs, PlayArgs};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const USAGE: &str = "usage:
  dzk play --decks-dir DIR[:DIR...] --pairs FILE.tsv --bot1 SPEC --bot2 SPEC --seed S --threads T --out games.jsonl
           [--n-pairs N] [--pair-range A:B] [--shard I/N] [--resume] [--data-out DIR]
           [--max-permanents N (default 100; 0 = none)] [--max-game-seconds S (default 0 = none)]
  dzk summarize GAMES.jsonl... [--out summary.json]
  dzk bench --decks-dir DIR[:DIR...] --pairs FILE.tsv --bot SPEC --seconds S --threads T1,T2,... [--seed S]
  dzk nnbench --decks-dir DIR[:DIR...] --pairs FILE.tsv [--net FILE.dzkn] [--bot SPEC] [--games N] [--seed S]
  dzk parity --net FILE.dzkn --data SHARD.dzd.gz --expected EXPECTED.json [--tol 1e-4]

bots: first | random[,mull=P] | greedy1[,mull=P] | mcts:N[,worlds=K][,c=C][,discount=D][,cache=M][,mull=P]
      | net:path=FILE[,t=T][,mull=P]
      | pmcts:N,net=FILE[,pt=T][,leaf=net|heuristic|mix:L][,noise=E][,alpha=A][,temp=T][,temp_moves=K][,fpu=R]
        + mcts options
      (heuristic@N = mcts:N; P = keep (default) | xmage | own)";

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.rest.iter().position(|a| a == flag)?;
        if i + 1 >= self.rest.len() {
            return None;
        }
        let v = self.rest.remove(i + 1);
        self.rest.remove(i);
        Some(v)
    }

    fn flag(&mut self, flag: &str) -> bool {
        match self.rest.iter().position(|a| a == flag) {
            Some(i) => {
                self.rest.remove(i);
                true
            }
            None => false,
        }
    }

    fn need(&mut self, flag: &str) -> Result<String, String> {
        self.take(flag).ok_or_else(|| format!("missing {flag}"))
    }

    fn finish(self) -> Result<(), String> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(format!("unexpected arguments: {:?}", self.rest))
        }
    }
}

fn num<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("bad value for {flag}: {v:?}"))
}

fn pair_of(flag: &str, v: &str, sep: char) -> Result<(usize, usize), String> {
    let (a, b) = v.split_once(sep).ok_or_else(|| format!("bad value for {flag}: {v:?}"))?;
    Ok((num(flag, a)?, num(flag, b)?))
}

fn main_inner() -> Result<(), String> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    let cmd = argv.remove(0);
    let mut a = Args { rest: argv };
    match cmd.as_str() {
        "play" => {
            let pa = PlayArgs {
                decks_dir: PathBuf::from(a.need("--decks-dir")?),
                pairs_file: PathBuf::from(a.need("--pairs")?),
                bot1: a.need("--bot1")?,
                bot2: a.need("--bot2")?,
                seed: num("--seed", &a.need("--seed")?)?,
                threads: num("--threads", &a.need("--threads")?)?,
                out: PathBuf::from(a.need("--out")?),
                n_pairs: a.take("--n-pairs").map(|v| num("--n-pairs", &v)).transpose()?,
                pair_range: a.take("--pair-range").map(|v| pair_of("--pair-range", &v, ':')).transpose()?,
                shard: a.take("--shard").map(|v| pair_of("--shard", &v, '/')).transpose()?,
                resume: a.flag("--resume"),
                data_out: a.take("--data-out").map(PathBuf::from),
                limits: GameLimits {
                    max_permanents: a
                        .take("--max-permanents")
                        .map(|v| num("--max-permanents", &v))
                        .transpose()?
                        .unwrap_or(dzk::record::DEFAULT_MAX_PERMANENTS),
                    max_seconds: a
                        .take("--max-game-seconds")
                        .map(|v| num("--max-game-seconds", &v))
                        .transpose()?
                        .unwrap_or(0.0),
                },
            };
            a.finish()?;
            if let Some((i, n)) = pa.shard {
                if n == 0 || i >= n {
                    return Err("--shard I/N needs 0 <= I < N".into());
                }
            }
            if pa.threads == 0 {
                return Err("--threads must be at least 1".into());
            }
            let s = run_play(&pa)?;
            let _ = writeln!(std::io::stdout(), "{}", serde_json::to_string_pretty(&s).unwrap());
        }
        "summarize" => {
            let out = a.take("--out").map(PathBuf::from);
            let files: Vec<PathBuf> = a.rest.drain(..).map(PathBuf::from).collect();
            if files.is_empty() || files.iter().any(|f| f.to_string_lossy().starts_with("--")) {
                return Err(format!("summarize needs games files\n{USAGE}"));
            }
            let s = serde_json::to_string_pretty(&run_summarize(&files)?).unwrap();
            match out {
                Some(p) => std::fs::write(&p, format!("{s}\n")).map_err(|e| format!("{}: {e}", p.display()))?,
                None => {
                    let _ = writeln!(std::io::stdout(), "{s}");
                }
            }
        }
        "bench" => {
            let threads: Vec<usize> =
                a.need("--threads")?.split(',').map(|t| num("--threads", t.trim())).collect::<Result<_, _>>()?;
            let ba = BenchArgs {
                decks_dir: PathBuf::from(a.need("--decks-dir")?),
                pairs_file: PathBuf::from(a.need("--pairs")?),
                bot: a.need("--bot")?,
                seconds: num("--seconds", &a.need("--seconds")?)?,
                threads,
                seed: a.take("--seed").map(|v| num("--seed", &v)).transpose()?.unwrap_or(1),
            };
            a.finish()?;
            if ba.threads.contains(&0) {
                return Err("--threads values must be at least 1".into());
            }
            let s = run_bench(&ba)?;
            let _ = writeln!(std::io::stdout(), "{}", serde_json::to_string_pretty(&s).unwrap());
        }
        "nnbench" => {
            let na = NnBenchArgs {
                decks_dir: PathBuf::from(a.need("--decks-dir")?),
                pairs_file: PathBuf::from(a.need("--pairs")?),
                net: a.take("--net").map(PathBuf::from),
                bot: a.take("--bot").unwrap_or_else(|| "random".into()),
                games: a.take("--games").map(|v| num("--games", &v)).transpose()?.unwrap_or(10),
                seed: a.take("--seed").map(|v| num("--seed", &v)).transpose()?.unwrap_or(1),
            };
            a.finish()?;
            let s = run_nnbench(&na)?;
            let _ = writeln!(std::io::stdout(), "{}", serde_json::to_string_pretty(&s).unwrap());
        }
        "parity" => {
            let net = PathBuf::from(a.need("--net")?);
            let data = PathBuf::from(a.need("--data")?);
            let expected = PathBuf::from(a.need("--expected")?);
            let tol: f64 = a.take("--tol").map(|v| num("--tol", &v)).transpose()?.unwrap_or(1e-4);
            a.finish()?;
            let s = run_parity(&net, &data, &expected)?;
            let _ = writeln!(std::io::stdout(), "{}", serde_json::to_string_pretty(&s).unwrap());
            let worst = s["max_abs_delta_logit"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                .max(s["max_abs_delta_value"].as_f64().unwrap_or(f64::INFINITY));
            if worst >= tol {
                return Err(format!("parity: max |delta| {worst:e} >= {tol:e}"));
            }
        }
        other => return Err(format!("unknown command {other:?}\n{USAGE}")),
    }
    Ok(())
}

fn main() -> ExitCode {
    match main_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dzk: {e}");
            ExitCode::from(2)
        }
    }
}
