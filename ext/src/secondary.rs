use crate::chain_complex::ChainComplex;
use crate::resolution::ResolutionInner;
use crate::CCC;
use algebra::combinatorics;
use algebra::milnor_algebra::{
    MilnorAlgebra as Algebra, MilnorBasisElement as MilnorElt, PPartMultiplier,
};
use algebra::module::homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism};
use algebra::module::FreeModule;
use algebra::module::Module;
use algebra::{Algebra as _, MilnorAlgebraT, SteenrodAlgebra};
use fp::prime::ValidPrime;
use fp::vector::{FpVector, FpVectorT};
use std::collections::HashMap;

type Resolution = ResolutionInner<CCC>;
type FMH = FreeModuleHomomorphism<FreeModule<SteenrodAlgebra>>;

const TWO: ValidPrime = ValidPrime::new(2);

pub fn compute_delta(res: &Resolution, max_s: u32, max_t: i32) -> Vec<FMH> {
    if max_s < 2 {
        return vec![];
    }
    let mut deltas = Vec::with_capacity(max_s as usize - 2);
    let delta2 = FreeModuleHomomorphism::new(res.module(2), res.module(0), 1);
    delta2.extend_by_zero_safe(max_t);
    deltas.push(delta2);

    let mut scratch = FpVector::new(TWO, 0);
    for s in 3..=max_s {
        let delta = FreeModuleHomomorphism::new(res.module(s), res.module(s - 2), 1);
        let d = res.differential(s - 2);
        let m = res.module(s);

        delta.extend_by_zero_safe(res.min_degree());
        for t in res.min_degree() + 1..=max_t {
            let num_gens = m.number_of_gens_in_degree(t);
            let target_dim = res.module(s - 2).dimension(t - 1);
            let mut results = vec![FpVector::new(TWO, target_dim); num_gens];

            scratch.set_scratch_vector_size(res.module(s - 3).dimension(t - 1));
            for (idx, result) in results.iter_mut().enumerate() {
                d_delta_g(res, s, t, idx, &mut scratch, &deltas[s as usize - 3]);
                d.quasi_inverse(t - 1).apply(result, 1, &scratch);
                scratch.set_to_zero_pure();
            }
            delta.add_generators_from_rows(&delta.lock(), t, results);
        }
        deltas.push(delta);
    }

    deltas
}

/// Computes d(delta(g));
pub fn d_delta_g(
    res: &Resolution,
    gen_s: u32,
    gen_t: i32,
    gen_idx: usize,
    result: &mut FpVector,
    prev_delta: &FMH,
) {
    let algebra = res.algebra();
    let algebra = algebra.milnor_algebra();

    let m = res.module(gen_s - 1);

    let d = res.differential(gen_s);
    let dg = d.output(gen_t, gen_idx);

    let mut op = MilnorElt::default();
    for (i, _) in dg.iter_nonzero() {
        let elt = m.index_to_op_gen(gen_t, i);
        algebra
            .basis_element_from_index(elt.operation_degree, elt.operation_index)
            .clone_into(&mut op);

        a_dd(
            res,
            &mut op,
            gen_s - 1,
            elt.generator_degree,
            elt.generator_index,
            result,
        );
    }

    if gen_s > 3 {
        debug_assert!(std::sync::Arc::ptr_eq(&prev_delta.source(), &d.target()));
        prev_delta.apply(result, 1, gen_t, dg);
    }

    #[cfg(debug_assertions)]
    if gen_s > 3 {
        let mut r = FpVector::new(TWO, res.module(gen_s - 4).dimension(gen_t - 1));
        res.differential(gen_s - 3)
            .apply(&mut r, 1, gen_t - 1, result);
        assert!(
            r.is_zero(),
            "dd != 0 at s = {}, t = {}, {}",
            gen_s,
            gen_t,
            r
        );
    }
}

/// Computes A(a, ddg)
pub fn a_dd(
    res: &Resolution,
    a: &mut MilnorElt,
    gen_s: u32,
    gen_t: i32,
    gen_idx: usize,
    result: &mut FpVector,
) {
    let target_deg = a.degree + gen_t - 1;

    let algebra = res.algebra();
    let algebra = algebra.milnor_algebra();

    let d = res.differential(gen_s);
    let dg = d.output(gen_t, gen_idx);
    let differential_l = res.differential(gen_s - 1);

    let module_h = res.module(gen_s - 1);
    let module_l = res.module(gen_s - 2);

    // (gen_t, gen_idx, target_element) -> coefficient
    let mut coefs: HashMap<(i32, usize, MilnorElt), u32> = HashMap::new();
    let mut temp = MilnorElt::default();

    let mut b = MilnorElt::default();
    let mut c = MilnorElt::default();
    for (i, _) in dg.iter_nonzero() {
        let elt = module_h.index_to_op_gen(gen_t, i);
        algebra
            .basis_element_from_index(elt.operation_degree, elt.operation_index)
            .clone_into(&mut b);

        let ddg = differential_l.output(elt.generator_degree, elt.generator_index);
        for (j, _) in ddg.iter_nonzero() {
            let elt2 = module_l.index_to_op_gen(elt.generator_degree, j);
            algebra
                .basis_element_from_index(elt2.operation_degree, elt2.operation_index)
                .clone_into(&mut c);

            let offset =
                module_l.generator_offset(target_deg, elt2.generator_degree, elt2.generator_index);
            let num_ops = algebra.dimension(a.degree + b.degree + c.degree - 1, 0);

            a_tau_y(
                algebra,
                a,
                &mut b,
                &mut c,
                &mut *result.borrow_slice(offset, offset + num_ops),
            );

            if a.p_part[0] > 0 {
                let mut multiplier = PPartMultiplier::<true>::new(TWO, &b.p_part, &c.p_part);
                temp.degree = b.degree + c.degree;
                while let Some(c_) = multiplier.next(&mut temp) {
                    let key = (elt2.generator_degree, elt2.generator_index, temp.clone());
                    let val = (c_ + coefs.get(&key).copied().unwrap_or(0)) % 4;
                    coefs.insert(key, val);
                }
            }
        }
    }
    if a.p_part[0] > 0 {
        a.p_part[0] -= 1;
        a.degree -= 1;
        for ((gen_t, gen_idx, elt), c) in coefs {
            if c == 0 {
                continue;
            }
            debug_assert_eq!(c, 2);

            let offset = module_l.generator_offset(a.degree + gen_t + elt.degree, gen_t, gen_idx);
            let num_ops = algebra.dimension(a.degree + elt.degree, 0);

            algebra.multiply(
                &mut *result.borrow_slice(offset, offset + num_ops),
                1,
                &a,
                &elt,
            );
        }
        a.p_part[0] += 1;
        a.degree += 1;
    }
}

macro_rules! sub {
    ($elt:ident, $k:expr, $n:expr) => {
        if $k > 0 {
            if $elt.p_part[$k - 1] < (1 << $n) {
                continue;
            }
            $elt.p_part[$k - 1] -= 1 << $n;
            $elt.degree -= combinatorics::xi_degrees(TWO)[$k - 1] * (1 << $n);
        }
    };
}
macro_rules! unsub {
    ($elt:ident, $k:expr, $n:expr) => {
        if $k > 0 {
            $elt.p_part[$k - 1] += 1 << $n;
            $elt.degree += combinatorics::xi_degrees(TWO)[$k - 1] * (1 << $n);
        }
    };
}

/// Compute the Y terms of A(a, τ(b, c))
fn a_tau_y(
    algebra: &Algebra,
    a: &mut MilnorElt,
    b: &mut MilnorElt,
    c: &mut MilnorElt,
    result: &mut FpVector,
) {
    let mut u = MilnorElt::default();

    // First compute τ(b, c)
    for k in 0..c.p_part.len() {
        sub!(c, k + 1, 0);
        for n in 1..b.p_part.len() + 1 {
            sub!(b, n, k);

            for m in 0..n {
                sub!(b, m, k);
                u.degree = b.degree + c.degree;

                let mut multiplier = PPartMultiplier::<false>::new(TWO, &b.p_part, &c.p_part);
                while multiplier.next(&mut u).is_some() {
                    a_y(algebra, a, m + k, n + k, &u, result);
                }
                unsub!(b, m, k);
            }
            unsub!(b, n, k);
        }
        unsub!(c, k + 1, 0);
    }
}

// Computes A(a, Y_{k, l} u)
fn a_y(
    algebra: &Algebra,
    a: &mut MilnorElt,
    k: usize,
    l: usize,
    u: &MilnorElt,
    result: &mut FpVector,
) {
    let mut rem = vec![];

    let mut temp = MilnorElt::default();
    let mut temp2 = MilnorElt {
        q_part: 0,
        p_part: vec![],
        degree: a.degree + u.degree + (1 << k) + (1 << l) - 2,
    };

    for i in 0..=a.p_part.len() {
        if i + k < l {
            continue;
        }

        sub!(a, i, k);
        for j in 0..=std::cmp::min(i + k - l, a.p_part.len()) {
            sub!(a, j, l);

            rem.clear();
            rem.resize(k + i, 0);

            rem[k + i - 1] += 1;
            rem[l + j - 1] += 1;

            debug_assert_eq!(
                temp2.degree,
                a.degree + u.degree + (1 << (k + i)) + (1 << (l + j)) - 2
            );
            let mut m = PPartMultiplier::<false>::new(TWO, &a.p_part, &u.p_part);
            while m.next(&mut temp).is_some() {
                let mut m2 = PPartMultiplier::<false>::new(TWO, &rem, &temp.p_part);
                while m2.next(&mut temp2).is_some() {
                    let idx = algebra.basis_element_to_index(&temp2);
                    result.add_basis_element(idx, 1);
                }
            }

            unsub!(a, j, l);
        }
        unsub!(a, i, k);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::construct_s_2;

    fn from_p_part(p_part: &[u32]) -> MilnorElt {
        let degree = p_part
            .iter()
            .enumerate()
            .map(|(i, &n)| combinatorics::xi_degrees(TWO)[i] * (n as i32))
            .sum();

        MilnorElt {
            q_part: 0,
            p_part: p_part.into(),
            degree,
        }
    }

    #[test]
    fn test_a_y() {
        let algebra = Algebra::new(TWO);

        let mut result = FpVector::new(TWO, 0);

        let mut check = |p_part: &[u32], k, l, u: &MilnorElt, ans: &str| {
            let mut a = from_p_part(p_part);

            let target_deg = a.degree + u.degree + (1 << k) + (1 << l) - 2;
            algebra.compute_basis(target_deg + 1);
            result.set_scratch_vector_size(algebra.dimension(target_deg, 0));
            a_y(&algebra, &mut a, k, l, u, &mut result);
            assert_eq!(
                &algebra.element_to_string(target_deg, &result),
                ans,
                "{} U_({},{})",
                a,
                k,
                l
            );
        };

        let e = MilnorElt::default();
        check(&[1], 0, 1, &e, "P(2)");
        check(&[1], 1, 2, &e, "0");
        check(&[0, 1], 0, 1, &e, "P(1, 1)");
        check(&[0, 2], 1, 3, &e, "P(0, 0, 2)");
        check(&[1, 2], 0, 1, &e, "P(2, 2)");
        check(&[1], 0, 1, &from_p_part(&[1]), "P(3) + P(0, 1)");
    }

    #[test]
    fn test_a_tau_y() {
        let algebra = Algebra::new(TWO);

        let mut result = FpVector::new(TWO, 0);

        let mut check = |a: &[u32], b: &[u32], c: &[u32], ans: &str| {
            let mut a = from_p_part(a);
            let mut b = from_p_part(b);
            let mut c = from_p_part(c);

            let target_deg = a.degree + b.degree + c.degree - 1;
            algebra.compute_basis(target_deg + 1);
            result.set_scratch_vector_size(algebra.dimension(target_deg, 0));
            a_tau_y(&algebra, &mut a, &mut b, &mut c, &mut result);
            assert_eq!(
                &algebra.element_to_string(target_deg, &result),
                ans,
                "A({}, τ({},{}))",
                a,
                b,
                c
            );
        };

        check(&[1], &[1], &[1], "P(2)");
        check(&[0, 2], &[0, 2], &[0, 2], "P(0, 1, 2)");
        check(&[0, 0, 4], &[0, 0, 4], &[0, 0, 4], "P(0, 0, 3, 0, 2)");
        check(&[1], &[2, 1], &[0, 1], "0");
        check(&[1], &[1], &[8], "P(6, 1)");
        check(&[1], &[2, 1], &[4], "0");
    }

    #[test]
    fn test_a_dd() {
        let bundle = construct_s_2("milnor");
        let resolution = &*bundle.resolution.read();

        let mut result = FpVector::new(TWO, 0);

        let mut check = |a: &[u32], gen_s: u32, gen_t: i32, gen_idx, ans: &str| {
            let mut a = from_p_part(a);

            let target_deg = a.degree + gen_t - 1;
            resolution.resolve_through_bidegree(gen_s, target_deg);
            let m = resolution.module(gen_s - 2);

            result.set_scratch_vector_size(m.dimension(target_deg));
            a_dd(
                &*resolution.inner,
                &mut a,
                gen_s,
                gen_t,
                gen_idx,
                &mut result,
            );
            assert_eq!(
                &m.element_to_string(target_deg, &result),
                ans,
                "A({}, dd x_({}, {}))",
                a,
                gen_t - gen_s as i32,
                gen_s
            );
        };

        check(&[1], 2, 5, 0, "P(5) x_{0,0} + P(2, 1) x_{0,0}");
        check(&[2], 2, 4, 0, "P(5) x_{0,0}");
        check(&[4], 2, 2, 0, "P(2, 1) x_{0,0}");
        check(
            &[1],
            3,
            10,
            0,
            "P(9) x_{1,0} + P(3, 2) x_{1,0} + P(2, 2) x_{2,0} + P(3, 1) x_{4,0} + P(0, 2) x_{4,0}",
        );
    }

    #[test]
    fn test_compute_differentials() {
        let mut result = String::new();
        let bundle = construct_s_2("milnor");
        let resolution = &*bundle.resolution.read();

        let s = 7;
        let t = 30;

        resolution.resolve_through_bidegree(s, t);
        let deltas = compute_delta(&resolution.inner, s, t);

        for s_ in 3..=s {
            for t_ in s_ as i32..=t {
                let module = resolution.module(s_);
                let module2 = resolution.module(s_ - 2);
                if module2.number_of_gens_in_degree(t_ - 1) == 0 {
                    continue;
                }

                let start = module2.generator_offset(t_ - 1, t_ - 1, 0);
                for idx in 0..module.number_of_gens_in_degree(t_) {
                    result.push_str(&format!(
                        "d_2* (x_({}, {})^({})]) = {:?}\n",
                        t_ - s_ as i32,
                        s_,
                        idx,
                        deltas[s_ as usize - 2]
                            .output(t_, idx)
                            .iter()
                            .skip(start)
                            .collect::<Vec<_>>()
                    ));
                }
            }
        }
        assert_eq!(
            result,
            r"d_2* (x_(0, 3)^(0)]) = [0]
d_2* (x_(14, 3)^(0)]) = [1]
d_2* (x_(7, 4)^(0)]) = [0]
d_2* (x_(14, 4)^(0)]) = [0]
d_2* (x_(15, 4)^(0)]) = [0]
d_2* (x_(17, 4)^(0)]) = [0]
d_2* (x_(14, 5)^(0)]) = [0]
d_2* (x_(17, 5)^(0)]) = [0]
d_2* (x_(18, 5)^(0)]) = [0]
d_2* (x_(20, 5)^(0)]) = [0]
d_2* (x_(14, 6)^(0)]) = [0]
d_2* (x_(16, 6)^(0)]) = [1]
d_2* (x_(17, 6)^(0)]) = [0, 1]
d_2* (x_(16, 7)^(0)]) = [0]
d_2* (x_(17, 7)^(0)]) = [1]
d_2* (x_(23, 7)^(0)]) = [0]
"
        );
    }
}