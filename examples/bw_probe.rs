//! External probe for biwhitening: runs the dense and CSR paths on a matrix
//! read from a COO text file and dumps the scalings plus diagnostics as JSON.
//!
//! Used to validate the crate from outside (independent reference check); not
//! part of the library API.
//!
//! Input format (whitespace separated):
//!   line 1: n p nnz
//!   then nnz lines: i j v      (sorted by i, then j; no duplicates)
//!
//! Usage: cargo run --release --example bw_probe -- in.coo out.json [max_iter] [tol]

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};

use faer::Mat;
use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::sparse::CsrMatrix;
use rmt_spca::spca::biwhitening_collapse;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bw_probe <in.coo> <out.json> [max_iter] [tol]");
        std::process::exit(2);
    }
    let max_iter: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let tol: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1e-5);

    let file = BufReader::new(fs::File::open(&args[1]).expect("open input"));
    let mut lines = file.lines().map(|l| l.expect("read line"));
    let header: Vec<usize> = lines
        .next()
        .expect("header")
        .split_whitespace()
        .map(|t| t.parse().expect("header int"))
        .collect();
    let (n, p, nnz) = (header[0], header[1], header[2]);

    let mut rows = vec![Vec::<(usize, f64)>::new(); n];
    let mut count = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let i: usize = it.next().unwrap().parse().unwrap();
        let j: usize = it.next().unwrap().parse().unwrap();
        let v: f64 = it.next().unwrap().parse().unwrap();
        rows[i].push((j, v));
        count += 1;
    }
    assert_eq!(count, nnz, "nnz mismatch");

    // CSR
    let mut indptr = Vec::with_capacity(n + 1);
    let (mut indices, mut data) = (Vec::with_capacity(nnz), Vec::with_capacity(nnz));
    indptr.push(0usize);
    for r in rows.iter() {
        for &(j, v) in r.iter() {
            indices.push(j);
            data.push(v);
        }
        indptr.push(indices.len());
    }
    let csr = CsrMatrix::new(n, p, indptr, indices, data).expect("valid CSR");

    // Dense copy of the same matrix
    let mut dense = Mat::<f64>::zeros(n, p);
    for (i, r) in rows.iter().enumerate() {
        for &(j, v) in r.iter() {
            dense.write(i, j, v);
        }
    }

    let bw = Biwhitener {
        max_iter,
        tol,
        damp: 1.0,
    };
    let t0 = std::time::Instant::now();
    let (cs, ds, it_s, conv_s, res_s) = bw.compute_csr(&csr);
    let sparse_secs = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (cd, dd, it_d, conv_d, res_d) = bw.compute(&dense);
    let dense_secs = t1.elapsed().as_secs_f64();

    let vec_of = |v: &faer::Col<f64>| (0..v.nrows()).map(|i| v.read(i)).collect::<Vec<f64>>();
    let (cs_v, ds_v, cd_v, dd_v) = (vec_of(&cs), vec_of(&ds), vec_of(&cd), vec_of(&dd));
    let (coll_s, ratio_s) = biwhitening_collapse(&cs_v, &ds_v);
    let (coll_d, ratio_d) = biwhitening_collapse(&cd_v, &dd_v);

    let arr = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.17e}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let json = format!(
        concat!(
            "{{\"n\":{},\"p\":{},\"nnz\":{},\"max_iter\":{},\"tol\":{},",
            "\"sparse\":{{\"iters\":{},\"converged\":{},\"residual\":{},\"secs\":{},",
            "\"collapsed_cells\":{},\"max_gene_factor_ratio\":{},\"c\":[{}],\"d\":[{}]}},",
            "\"dense\":{{\"iters\":{},\"converged\":{},\"residual\":{},\"secs\":{},",
            "\"collapsed_cells\":{},\"max_gene_factor_ratio\":{},\"c\":[{}],\"d\":[{}]}}}}"
        ),
        n,
        p,
        nnz,
        max_iter,
        tol,
        it_s,
        conv_s,
        res_s,
        sparse_secs,
        coll_s,
        ratio_s,
        arr(&cs_v),
        arr(&ds_v),
        it_d,
        conv_d,
        res_d,
        dense_secs,
        coll_d,
        ratio_d,
        arr(&cd_v),
        arr(&dd_v),
    );
    fs::write(&args[2], json).expect("write output");
    eprintln!(
        "sparse: {it_s} iters conv={conv_s} res={res_s:.3e} {sparse_secs:.2}s | \
         dense: {it_d} iters conv={conv_d} res={res_d:.3e} {dense_secs:.2}s"
    );
}
