fn main() {
    eprintln!("faer parallelism: {:?}", faer::get_global_parallelism());
    eprintln!("rayon threads: {}", rayon::current_num_threads());

    let n = 1000_usize;
    let p = 5000_usize;
    let x = faer::Mat::<f64>::from_fn(n, p, |i, j| (i * 7 + j * 13) as f64 * 0.001);

    let t = std::time::Instant::now();
    let _s: faer::Mat<f64> = x.as_ref().transpose() * x.as_ref();
    eprintln!("5000x5000 GEMM (parallel): {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    faer::set_global_parallelism(faer::Parallelism::None);
    let t = std::time::Instant::now();
    let _s: faer::Mat<f64> = x.as_ref().transpose() * x.as_ref();
    eprintln!("5000x5000 GEMM (serial):   {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
}
