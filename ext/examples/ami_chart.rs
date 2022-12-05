use algebra::{module::homomorphism::ModuleHomomorphism, AlgebraType};
use anyhow::{Error, Result};
use chart::{Backend, Orientation, SvgBackend};
#[cfg(not(feature = "nassau"))]
use clap::{error::ErrorKind, CommandFactory};
use clap::{ArgGroup, Parser};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    utils,
    utils::Timer,
};
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use serde_json::json;
use std::collections::HashMap;
use std::io;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Arc;
// Calling compute_through_stem for different modules within a rayon scope seemed to result in
// deadlocks, so we will do some manual thread management.
// As compute_through_stem is already parallelized, spawning new threads for different k's
// brings performance benefits only on machines with many cores.
use std::thread;

#[derive(Parser)]
#[command(about, version)]
#[command(group(ArgGroup::new("limit").multiple(true).required(true).args(["k_max", "n_max"])))]
struct Cli {
    /// Try to detect classes in RP_-k_inf's up to k=K_MAX
    /// [positive, default: 2*N_MAX+1]
    #[arg(short, long, group = "limit")]
    k_max: Option<NonZeroU32>,
    /// Limit the stem of the classes whose Mahowald invariants are computed
    /// [positive, default: K_MAX-1]
    #[arg(short, long, group = "limit")]
    n_max: Option<NonZeroU32>,
    /// Limit the filtration of the classes whose Mahowald invariants are computed
    /// [positive, default: (N_MAX+1)/2+1]
    #[arg(short, long)]
    s_max: Option<NonZeroU32>,
    /// Go T_EXTENSION steps further in the t direction beyond the n=N_MAX line
    #[arg(short, long, default_value_t = 0)]
    t_extension: u32,

    /// Save directory for S_2
    #[arg(long)]
    s_2_path: Option<PathBuf>,
    /// Load the quasi-inverses of the differentials in the resolution for S_2 lazily (requires
    /// S_2_PATH)
    #[cfg(not(feature = "nassau"))]
    #[arg(long, short)]
    lazy_quasi_inverses: bool,
    /// Directory containing save directories for RP_-k_inf's
    #[arg(long)]
    p_k_prefix: Option<PathBuf>,
    /// Process N RP_-k_inf's in parallel
    #[arg(long, short, value_name="N", default_value_t = NonZeroUsize::new(1).unwrap())]
    parallel_ks: NonZeroUsize,
}

// TODO: compute s_max based on the loaded resolution of S_2
fn s_max_for_n(n: u32) -> u32 {
    ((n + 1) / 2) + 1
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // See further below for visualizations of the ranges in which we compute the Mahowald
    // invariants depending on what of k_max, n_max_global, s_max_global and t_extension are.
    let (k_max, mut n_max_global) = match (cli.k_max.map(|k| k.get()), cli.n_max.map(|n| n.get())) {
        (Some(k), Some(n)) => (k, n),
        (Some(k), _) => (k, k - 1),
        (_, Some(n)) => (2 * n + 1, n),
        (None, None) => unreachable!(),
    };
    if k_max - 1 < n_max_global {
        // A larger value doesn't make sense because a class in stem n isn't detected before k=n+1.
        n_max_global = k_max - 1;
        eprint!("Warning: Will only compute through n={n_max_global}, ");
        eprintln!("for higher stems K_MAX must be increased.");
    }
    let s_threshold = s_max_for_n(n_max_global);
    let s_max_global = if let Some(s) = cli.s_max {
        let s = s.get();
        if s <= s_threshold {
            s
        } else {
            eprint!("Warning: Will only compute up to s={s_threshold}, there are no ");
            eprintln!("classes of higher filtration in the first {n_max_global} positive stem(s).");
            s_threshold
        }
    } else {
        s_threshold
    };
    let n_max_global = n_max_global as i32;
    let t_extension = if cli.t_extension > s_max_global {
        eprintln!("Warning: Truncating T_EXTENSION to S_MAX={s_max_global}.");
        s_max_global as i32
    } else {
        cli.t_extension as i32
    };

    #[cfg(not(feature = "nassau"))]
    if cli.lazy_quasi_inverses && cli.s_2_path.is_none() {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "S_2_PATH is required to load quasi-inverses lazily",
            )
            .exit();
    }
    let parallel_ks = cli.parallel_ks.get();

    // We precompute/preload a resolution of F_2 in a range that will be required for further
    // computations.
    // It could be computed/loaded "lazily" at each k, but this way we don't have to think about
    // it in the "main loop" and for large computations we want to use a precomputed resolution of
    // F_2 anyway.
    let s_2_config = json!({
        "p": 2,
        "type": "finite dimensional module",
        "gens": { "x0": 0 },
        "actions": [],
    });
    #[cfg(not(feature = "nassau"))]
    let s_2_resolution = Arc::new({
        let mut resolution = utils::construct((s_2_config, AlgebraType::Milnor), cli.s_2_path)?;
        resolution.load_quasi_inverse = !cli.lazy_quasi_inverses;
        resolution
    });
    #[cfg(feature = "nassau")]
    let s_2_resolution = Arc::new(utils::construct(
        (s_2_config, AlgebraType::Milnor),
        cli.s_2_path,
    )?);
    let timer = Timer::start();
    // We "micromanage" which bidegrees are resolved because computing resolutions is the main
    // bottleneck.
    // (This is of course less relevant at this point if we use a precomputed resolution of F_2,
    // but the argument applies to RP_-k_inf's too.)
    // We first compute the resolution through n=(k_max-1)+n_max which will be needed in all cases.
    s_2_resolution.compute_through_stem(s_max_global, k_max as i32 - 1 + n_max_global);
    if t_extension > 0 {
        if n_max_global + t_extension <= k_max as i32 - 1 {
            // For k large enough the resolved area for RP_-k_inf will look as follows.
            // (n+1)/2+1 ---
            //           **|
            //           ***\
            //           ****\
            //           *****\
            //           **|***\     |
            //            n-1 n-1+t k-2
            let t_max = k_max as i32 - 1 + n_max_global + t_extension;
            s_2_resolution.compute_through_bidegree(s_max_global, t_max);
        } else {
            // For k large enough the resolved area for RP_-k_inf will look as follows.
            // (n+1)/2+1 ---
            //           **|
            //           ***\
            //           ****\
            //           *****\
            //           ******\
            //           ******|\
            //           ******| \
            //           ******|  \
            //           **|***|   \
            //            n-1 k-2 n-1+t
            let n_start = 2 * k_max as i32 - 2;
            let s_start = n_max_global as u32 + t_extension as u32 - k_max + 1;
            let diff = k_max - 1 - n_max_global as u32;
            for i in 0..=diff {
                s_2_resolution.compute_through_stem(s_start + i, n_start - i as i32);
            }
        }
    }
    timer.end(format_args!("Computed/loaded the resolution for S_2."));

    let cloned_s_2_resolution = s_2_resolution.clone();
    // We will handle the single-threaded and the multi-threaded cases differently, so we create
    // this closure which will be used at each step in both cases.
    let process_k = move |k: u32| {
        let module_name = format!("RP_-{k}_inf");

        let p_k_config = json! ({
            "p": 2,
            "type": "real projective space",
            "min": -(k as i32),
        });
        let mut p_k_path = cli.p_k_prefix.clone();
        if let Some(p) = p_k_path.as_mut() {
            p.push(PathBuf::from(&module_name))
        };
        #[cfg(not(feature = "nassau"))]
        let p_k_resolution = Arc::new({
            let mut resolution = utils::construct((p_k_config, AlgebraType::Milnor), p_k_path)?;
            // Looks like we don't need quasi-inverses for resolutions of RP_-k_inf's...
            resolution.load_quasi_inverse = false;
            resolution
        });
        #[cfg(feature = "nassau")]
        let p_k_resolution = Arc::new(utils::construct(
            (p_k_config, AlgebraType::Milnor),
            p_k_path,
        )?);
        let timer = Timer::start();
        if k as i32 - 1 <= n_max_global {
            // In this case, the resolved area will look as follows.
            // (k+1)/2+1 ---
            //           **|
            //           **|   |
            //            k-2 n-1
            let n_max = k as i32 - 2;
            let s_max = s_max_for_n(k - 1).min(s_max_global);
            p_k_resolution.compute_through_stem(s_max, n_max);
        } else {
            let n_max = n_max_global - 1;
            let s_max = s_max_global;
            // For the area resolved the case of a "t extension", see the pictures for S_2.
            if t_extension > 0 {
                if n_max_global + t_extension <= k as i32 - 1 {
                    let t_max = n_max + t_extension;
                    p_k_resolution.compute_through_bidegree(s_max, t_max);
                } else {
                    let n_start = k as i32 - 2;
                    let s_start = n_max_global as u32 + t_extension as u32 - k + 1;
                    let diff = k - 1 - n_max_global as u32;
                    for i in 0..=diff {
                        p_k_resolution.compute_through_stem(s_start + i, n_start - i as i32);
                    }
                }
            }
            // In this case, the following area will need to be resolved regardless of
            // "t extension".
            // (n+1)/2+1 ---
            //           **|
            //           **|   |
            //            n-1 k-2
            p_k_resolution.compute_through_stem(s_max, n_max);
        };
        timer.end(format_args!(
            "Computed/loaded the resolution for {module_name}."
        ));

        let timer = Timer::start();
        let bottom_cell = ResolutionHomomorphism::from_class(
            String::from("bottom_cell"),
            p_k_resolution.clone(),
            s_2_resolution.clone(),
            0,
            -(k as i32),
            &[1],
        );
        bottom_cell.extend_all();
        timer.end(format_args!(
            "Computed/loaded the bottom cell map for {module_name}."
        ));

        let timer = Timer::start();
        let minus_one_cell = ResolutionHomomorphism::from_class(
            String::from("minus_one_cell"),
            p_k_resolution.clone(),
            s_2_resolution.clone(),
            0,
            -1,
            &[1],
        );
        minus_one_cell.extend_all();
        timer.end(format_args!(
            "Computed/loaded the (-1)-cell map for {module_name}."
        ));

        let timer = Timer::start();
        // We could probably avoid allocating a vector for each k, but these allocations are very
        // small compared to resolution and homomorphism data, so their effect should be small.
        let mut mahowald_invariants = Vec::new();
        for (s, _, t) in s_2_resolution
            .iter_stem()
            .filter(|&(s, n, t)| 1 <= n && p_k_resolution.has_computed_bidegree(s, t - 1))
        {
            let t_bottom = t + k as i32 - 1;
            let bottom_s_2_gens = s_2_resolution.module(s).number_of_gens_in_degree(t_bottom);
            let minus_one_s_2_gens = s_2_resolution.module(s).number_of_gens_in_degree(t);
            let t_p_k = t - 1;
            let p_k_gens = p_k_resolution.module(s).number_of_gens_in_degree(t_p_k);
            if bottom_s_2_gens > 0 && minus_one_s_2_gens > 0 && p_k_gens > 0 {
                // Since we will try to find preimages of some elements under the bottom cell map,
                // it will be useful to extract its matrix representation at the current bidegree.
                // (For readers unfamiliar with the sseq library it should be noted that sseq works
                // with row vectors and matrices act on vectors from the right.)
                let bottom_cell_map = bottom_cell.get_map(s);
                let mut matrix = vec![vec![0; p_k_gens]; bottom_s_2_gens];
                for p_k_gen in 0..p_k_gens {
                    let output = bottom_cell_map.output(t_p_k, p_k_gen);
                    for (s_2_gen, row) in matrix.iter_mut().enumerate() {
                        let index = bottom_cell_map
                            .target()
                            .operation_generator_to_index(0, 0, t_bottom, s_2_gen);
                        row[p_k_gen] = output.entry(index);
                    }
                }

                let (padded_columns, mut matrix) = Matrix::augmented_from_vec(TWO, &matrix);
                let rank = matrix.row_reduce();
                // It makes sense to look for Mahowald invariants only if there are non-zero
                // elements in the image of the bottom cell map.
                if rank > 0 {
                    let f2_vec_to_gen_list = |v: &FpVector| {
                        v.iter()
                            .enumerate()
                            .filter_map(|(i, e)| if e == 1 { Some((s, t_bottom, i)) } else { None })
                            .collect::<Vec<_>>()
                    };
                    let indeterminacy: Vec<_> = matrix
                        .compute_kernel(padded_columns)
                        .basis()
                        .iter()
                        .map(f2_vec_to_gen_list)
                        .collect();
                    let image_subspace = matrix.compute_image(p_k_gens, padded_columns);
                    // The "quasiinverse" can be used to compute a preimage for elements in the image.
                    let quasi_inverse = matrix.compute_quasi_inverse(p_k_gens, padded_columns);

                    for i in 0..minus_one_s_2_gens {
                        let mut image = FpVector::new(TWO, p_k_gens);
                        minus_one_cell.act(image.as_slice_mut(), 1, s, t, i);
                        if !image.is_zero() && image_subspace.contains(image.as_slice()) {
                            let mut mahowald_invariant = FpVector::new(TWO, bottom_s_2_gens);
                            quasi_inverse.apply(
                                mahowald_invariant.as_slice_mut(),
                                1,
                                image.as_slice(),
                            );
                            mahowald_invariants.push((
                                s,
                                t,
                                i,
                                k,
                                f2_vec_to_gen_list(&mahowald_invariant),
                                indeterminacy.clone(),
                            ));
                        }
                    }
                }
            }
        }
        let registered = mahowald_invariants.len();
        timer.end(format_args!(
            "Registered {registered} Mahowald invariant(s) at k={k}."
        ));
        Ok::<_, Error>(mahowald_invariants)
    };

    let mut mahowald_invariants = Vec::new();
    if parallel_ks == 1 {
        for k in 3..=k_max {
            mahowald_invariants.append(&mut process_k(k)?);
        }
    } else if k_max >= 3 {
        // We start at k_max because larger k's tend to take more time.
        let mut next_k = k_max;
        let mut handles = Vec::new();
        let process_k = Arc::new(process_k);
        for _ in 0..(parallel_ks.min(next_k as usize - 2)) {
            let process_k = process_k.clone();
            handles.push(Some((next_k, thread::spawn(move || process_k(next_k)))));
            next_k -= 1;
        }
        while handles.iter().any(|h| h.is_some()) {
            for outer in handles.iter_mut() {
                if let Some((_, handle)) = outer {
                    if handle.is_finished() {
                        let (k, handle) = outer.take().unwrap();
                        let mut v = handle
                            .join()
                            .map_err(|_| Error::msg(format!("Thread for k={k} panicked")))??;
                        if next_k >= 3 {
                            let process_k = process_k.clone();
                            *outer = Some((next_k, thread::spawn(move || process_k(next_k))));
                            next_k -= 1;
                        }
                        mahowald_invariants.append(&mut v);
                    }
                }
            }
        }
    };

    println!("<!DOCTYPE html>");
    println!("<html>");
    println!("<head>");
    println!("<title>Some algebraic Mahowald invariants</title>");
    println!("<style>table, th, td {{border: 1px solid black; border-collapse: collapse; padding:0.5rem}}</style>");
    println!("<style>html {{display: table; margin: auto;}} body {{display: table-cell; vertical-align: middle}} table {{margin: 0 auto}} h1, div {{text-align: center}}</style>");
    println!("</head>");
    println!("<body>");
    println!("<h1>Some algebraic Mahowald invariants</h1>");
    let mut backend = SvgBackend::new(io::stdout());

    let s_2_resolution = cloned_s_2_resolution;
    let sseq = s_2_resolution.to_sseq();
    let products = [
        (
            String::from("h0"),
            s_2_resolution.filtration_one_products(1, 0),
        ),
        (
            String::from("h1"),
            s_2_resolution.filtration_one_products(2, 0),
        ),
    ];
    sseq.write_to_graph(&mut backend, 2, false, products.iter(), |_| Ok(()))?;

    println!(r#"<style>.mi {{visibility: hidden}} </style>"#);
    println!(
        r#"<defs><filter x="0" y="0" width="1" height="1" id="label_filter"><feFlood flood-color="white" result="bg" /><feMerge><feMergeNode in="bg"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>"#
    );
    let (_, _, y_zero) = backend.get_coords(0, 0, 0);
    let (_, _, y_one) = backend.get_coords(0, 1, 0);
    let y_diff = y_one - y_zero;

    let mut gens_by_n = vec![vec![]; n_max_global as usize + 1];
    for (s, n, t) in s_2_resolution.iter_stem() {
        if n > 0 && n <= n_max_global {
            for i in 0..s_2_resolution.module(s).number_of_gens_in_degree(t) {
                gens_by_n[n as usize].push((s, t, i));
            }
        }
    }
    backend.text(
        0,
        s_max_global as i32,
        r#"<tspan y="1em" style="font-size:0.5rem">n:</tspan>"#,
        Orientation::Below,
    )?;
    backend.text(
        0,
        s_max_global as i32,
        r#"<tspan y="2.25em" style="fill: blue; font-size: 0.5rem; font-weight:bold;">Max k:</tspan>"#,
        Orientation::Below,
    )?;
    let hit_map: HashMap<_, _> = mahowald_invariants
        .iter()
        .map(|&(s, t, i, k, _, _)| ((s, t, i), k))
        .collect();
    for n in 1..=n_max_global {
        let k_max_n = gens_by_n[n as usize]
            .iter()
            .try_fold(0, |acc, k| hit_map.get(k).map(|&k| acc.max(k)));
        let label = if let Some(k) = k_max_n {
            format!("{k}")
        } else {
            String::from("??")
        };
        backend.text(
            n,
            s_max_global as i32,
            format!(r#"<tspan y="1em" style="font-size:0.5rem">{n}</tspan>"#),
            Orientation::Below,
        )?;
        backend.text(
            n,
            s_max_global as i32,
            format!(r#"<tspan y="2.25em" style="fill: blue; font-size: 0.5rem; font-weight:bold;">{label}</tspan>"#),
            Orientation::Below,
        )?;
    }

    let display_vector = |v: &Vec<(u32, i32, usize)>| {
        if v.is_empty() {
            String::from("0")
        } else {
            v.iter()
                .map(|x| format!("{:?}", x))
                .collect::<Vec<_>>()
                .join(" + ")
        }
    };
    mahowald_invariants.sort_unstable_by_key(|&(s, t, i, _, _, _)| (s as i32 - t, s, i));
    for (s, t, i, k, mi, _) in mahowald_invariants.iter() {
        let (s, t, i, k) = (*s, *t, *i, *k);

        let n = t - s as i32;
        println!(r#"<g id="mi-group-{s}-{t}-{i}">"#);

        let red = 255 * (k as usize - n as usize - 1) / n as usize;
        if red > 255 {
            panic!("Did ABLC just fail?");
        }
        let color = format!("#{red:02X}40B0");
        let (r, mut x, mut y) = backend.get_coords(n, s as i32, i);
        //println!("<a href=\"#mi-group-{s}-{t}\">");
        println!(r#"<circle cx="{x}" cy="{y}" r="{r}" style="fill:{color}" id="{s}-{t}-{i}"/>"#);
        //println!("</a>");

        x += r;
        y -= r;
        let mut x_m = 0.0;
        for &(s_m, t_m, i_m) in mi.iter() {
            let (r_current, x_current, y_current) =
                backend.get_coords(t_m - s_m as i32, s_m as i32, i_m);
            println!(
                r#"<circle cx="{x_current}" cy="{y_current}" r="{r_current}" style="fill:blue" class="mi mi-{s}-{t}-{i}"/>"#
            );
            x_m += x_current;
        }
        x_m /= mi.len() as f32;
        x_m -= r;
        let y_mid = y + y_diff;
        let x_mid = x + (x_m - x) / 2.0;
        println!(
            r#"<path d="M {x} {y} Q {x_mid} {y_mid} {x_m} {y}" stroke="{color}" fill="transparent" class="mi mi-{s}-{t}-{i}"/>"#
        );

        let mi = display_vector(mi);
        let label = format!("M(({s}, {t}, {i})) = {mi}");
        println!(
            r#"<text x="{x_mid}" y="{y_mid}" text-anchor="middle" dominant-baseline="text-top" style="fill:blue" class="mi mi-{s}-{t}-{i}" id="mi-label-{s}-{t}-{i}">{label}</text>"#,
        );

        println!("</g>");
        println!(
            r#"<style>#mi-group-{s}-{t}-{i}:hover .mi-{s}-{t}-{i} {{visibility: visible}} </style>"#
        );
        println!(
            r#"<style>#mi-group-{s}-{t}-{i}:hover #mi-label-{s}-{t}-{i} {{filter: url(#label_filter)}} </style>"#
        );
        println!(
            r#"<style>#mi-group-{s}-{t}-{i}:target .mi-{s}-{t}-{i} {{visibility: visible}} </style>"#
        );
        println!(
            r#"<style>#mi-group-{s}-{t}-{i}:target #mi-label-{s}-{t}-{i} {{filter: url(#label_filter)}} </style>"#
        );
    }
    drop(backend);

    mahowald_invariants.sort_unstable_by_key(|&(s, t, i, _, _, _)| (t - s as i32, s, i));
    println!("<table>");
    println!("<tr><th></th><th>n</th><th>s</th><th>t</th><th>i</th><th>k</th><th>M(-)</th><th>indeterminacy</th><tr>");
    for (s, t, i, k, mi, indet) in mahowald_invariants {
        let n = t as u32 - s;
        let mi = display_vector(&mi);
        let indet = if indet.is_empty() {
            String::from("")
        } else {
            format!(
                "{{{}}}",
                indet
                    .iter()
                    .map(display_vector)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        println!("<tr><td><a href=\"#mi-group-{s}-{t}-{i}\">show</a></td><td>{n}</td><td>{s}</td><td>{t}</td><td>{i}</td><td>{k}</td><td>{mi}</td><td>{indet}</td><tr>");
    }
    println!("</table>");

    println!(
        r#"<div>Computed and rendered using the <a href="https://github.com/spectralsequences/sseq">sseq</a> library.</div>"#
    );
    println!("</body>");
    println!("</html>");

    Ok(())
}
