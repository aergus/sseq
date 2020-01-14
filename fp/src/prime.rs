pub const MAX_PRIME_INDEX : usize = 54;
pub const MAX_PRIME : usize = 251;
const NOT_A_PRIME : usize = !1;
pub const MAX_MULTINOMIAL_LEN : usize = 10;
pub const PRIME_TO_INDEX_MAP : [usize; MAX_PRIME+1] = [
    !1, !1, 0, 1, !1, 2, !1, 3, !1, !1, !1, 4, !1, 5, !1, !1, !1, 6, !1,
    7,  !1, !1, !1, 8, !1, !1, !1, !1, !1, 9, !1, 10, !1, !1, !1, !1, !1,
    11, !1, !1, !1, 12, !1, 13, !1, !1, !1, 14, !1, !1, !1, !1, !1, 15,
    !1, !1, !1, !1, !1, 16, !1, 17, !1, !1, !1, !1, !1, 18, !1, !1, !1,
    19, !1, 20, !1, !1, !1, !1, !1, 21, !1, !1, !1, 22, !1, !1, !1, !1,
    !1, 23, !1, !1, !1, !1, !1, !1, !1, 24, !1, !1, !1, 25, !1, 26, !1,
    !1, !1, 27, !1, 28, !1, !1, !1, 29, !1, !1, !1, !1, !1, !1, !1, !1,
    !1, !1, !1, !1, !1, 30, !1, !1, !1, 31, !1, !1, !1, !1, !1, 32, !1,
    33, !1, !1, !1, !1, !1, !1, !1, !1, !1, 34, !1, 35, !1, !1, !1, !1,
    !1, 36, !1, !1, !1, !1, !1, 37, !1, !1, !1, 38, !1, !1, !1, !1, !1,
    39, !1, !1, !1, !1, !1, 40, !1, 41, !1, !1, !1, !1, !1, !1, !1, !1,
    !1, 42, !1, 43, !1, !1, !1, 44, !1, 45, !1, !1, !1, !1, !1, !1, !1,
    !1, !1, !1, !1, 46, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, 47,
    !1, !1, !1, 48, !1, 49, !1, !1, !1, 50, !1, !1, !1, !1, !1, 51, !1,
    52, !1, !1, !1, !1, !1, !1, !1, !1, !1, 53 //, !1, !1, !1, !1
];

pub fn is_valid_prime(p : u32) -> bool {
    (p as usize) < MAX_PRIME && PRIME_TO_INDEX_MAP[p as usize] != NOT_A_PRIME
}

// Mathematica:
// "[\n    " <> # <> "\n]" &[
//  StringJoin @@
//   StringReplace[
//    ToString /@
//     Riffle[Map[If[# > 2^31, 0, #] &,
//       Function[p,
//         PadRight[PowerMod[#, -1, p] & /@ Range[p - 1],
//          Prime[8] - 1]] /@ Prime[Range[8]], {2}], ",\n    "], {"{" ->
//      "[", "}" -> "]"}]]
static INVERSE_TABLE : [[u32; 19]; 8] = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 3, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 4, 5, 2, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 6, 4, 3, 9, 2, 8, 7, 5, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 7, 9, 10, 8, 11, 2, 5, 3, 4, 6, 12, 0, 0, 0, 0, 0, 0],
    [0, 1, 9, 6, 13, 7, 3, 5, 15, 2, 12, 14, 10, 4, 11, 8, 16, 0, 0],
    [0, 1, 10, 13, 5, 4, 16, 11, 12, 17, 2, 7, 8, 3, 15, 14, 6, 9, 18]
];

// Uses a the lookup table we initialized.
pub fn inverse(p : u32, k : u32) -> u32{
    assert!(is_valid_prime(p));
    let result = INVERSE_TABLE[PRIME_TO_INDEX_MAP[p as usize]][k as usize];
    assert!(result != 0);
    result
}

pub fn minus_one_to_the_n(p : u32, i : u32) -> u32 {
    if i % 2 == 0 { 1 } else { p - 1 }
}

// Makes a lookup table for n choose k when n and k are both less than p.
// Lucas's theorem reduces general binomial coefficients to this case.

/// Calling this function safely requires that
///  * `p` is a valid prime
///  * `p <= 19`
///  * `k, n < p`.
/// These invariants are often known apriori because k and n are obtained by reducing mod p (and
/// the first two are checked beforehand), so it is better to expose an unsafe interface that
/// avoids these checks.
unsafe fn direct_binomial(p : u32, n : u32, k : u32) -> u32{
    *BINOMIAL_TABLE.get_unchecked(*PRIME_TO_INDEX_MAP.get_unchecked(p as usize)).get_unchecked(n as usize).get_unchecked(k as usize)
}


/// Computes b^e.
pub fn integer_power(b : u32, e : u32) -> u32 {
    let mut b = b;
    let mut e = e;
    let mut result = 1u32;
    while e > 0 {
        if e&1 == 1 {
            result *= b;
        }
        b *= b;
        e >>= 1;
    }
    result
}

/// Compute b^e mod p.
/// We use this for computing modulo inverses.
pub fn power_mod(p : u32, b : u32, e : u32) -> u32{
    let mut b = b;
    let mut e = e;
    let mut result = 1u32;
//      b is b^{2^i} mod p
//      if the current bit of e is odd, mutliply b^{2^i} mod p into r.
    while e > 0 {
        if (e&1) == 1 {
            result = (result*b)%p;
        }
        b = (b*b)%p;
        e >>= 1;
    }
    result
}

// Discrete log base p of n.
pub fn logp(p : u32, mut n : u32) -> u32 {
    let mut result = 0u32;
    while n > 0 {
        n /= p;
        result += 1;
    }
    result
}

//Multinomial coefficient of the list l
fn multinomial2(l : &mut [u32]) -> u32 {
    let mut bit_or = 0u32;
    let mut sum = 0u32;
    for &mut e in l {
        sum += e;
        bit_or |= e;
//        if(bit_or < sum){
//            return 0;
//        }
    }
    if bit_or == sum { 1 } else { 0 }
}

//Mod 2 binomial coefficient n choose k
fn binomial2(n : i32, k : i32) -> u32 {
    if n < k {
        0
    } else if (n-k) & k == 0 {
        1
    } else {
        0
    }
}

//Mod p multinomial coefficient of l. If p is 2, more efficient to use Multinomial2.
//This uses Lucas's theorem to reduce to n choose k for n, k < p.
fn multinomial_odd(p : u32, l : &mut [u32]) -> u32 {
    assert!(p > 0 && p < 20 && PRIME_TO_INDEX_MAP[p as usize] != NOT_A_PRIME);
    let mut n : u32 = l.iter().sum();
    if n == 0 {
        return 1;
    }
    let mut answer : u32 = 1;

    while n > 0 {
        let mut multi : u32 = 1;

        let total_entry = n % p;
        n /= p;

        let mut partial_sum : u32 = l[0] % p;
        l[0] /= p;

        for ll in l.iter_mut().skip(1) {
            let entry = *ll % p;
            *ll /= p;

            partial_sum += entry;
            if partial_sum > total_entry {
                // This early return is necessary because direct_binomial only works when
                // partial_sum < 19
                return 0;
            }
            // This is safe because p < 20, partial_sum <= total_entry < p and entry < p.
            multi *= unsafe { direct_binomial(p, partial_sum, entry) };
            multi %= p;
        }
        answer *= multi;
        answer %= p;
    }
    answer
}

//Mod p binomial coefficient n choose k. If p is 2, more efficient to use Binomial2.
fn binomial_odd(p : u32, n : i32, k : i32) -> u32 {
    // When computing mod p, we check if p is zero. The PRIME_TO_INDEX_MAP check ensures that p !=
    // 0, but the compiler does not know that, so repeats the check in the loop every time.
    assert!(p > 0 && p < 20 && PRIME_TO_INDEX_MAP[p as usize] != NOT_A_PRIME);
    if n < k || k < 0 {
        return 0;
    }

    let mut k = k as u32;
    let mut n = n as u32;

    let mut answer : u32 = 1;

    while n > 0 {
        // This is safe because p < 20 and anything mod p is < p.
        answer *= unsafe { direct_binomial(p, n % p, k % p) };
        answer %= p;
        n /= p;
        k /= p;
    }
    answer
}

/// This computes the multinomial coefficient $\binom{n}{l_1 \ldots l_k}\bmod p$, where $n$
/// is the sum of the entries of l. This function modifies the entries of l.
pub fn multinomial(p : u32, l : &mut [u32]) -> u32 {
    if p == 2 {
        multinomial2(l)
    } else {
        multinomial_odd(p, l)
    }
}

//Dispatch to Binomial2 or BinomialOdd
pub fn binomial(p : u32, n : i32, k : i32) -> u32 {
    if p == 2{
        binomial2(n, k)
    } else {
        binomial_odd(p, n, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn direct_binomial(t *testing.T) {
    //     tables := []struct {
    //         n int
    //         k int
    //         p int
    //         output int
    //     }{
    //         {21, 2, 23, 210},
    //         {13, 9, 23, 715},
    //         {12, 8, 23, 495},
    //         {13, 8, 23, 1287},
    //         {14, 8, 23, 3003},
    //         {14, 9, 23, 2002},
    //         {15, 5, 23, 3003},
    //         {15, 8, 23, 6435},
    //         {15, 9, 23, 5005},
    //         {16, 9, 23, 11440},
    //     }
    //     for _, table := range tables {
    //         output := direct_binomial(table.n, table.k, table.p)
    //         if output != table.output  % table.p {
    //             t.Errorf("Ran directBinomial(%v,%v) expected %v, got %v", table.n, table.k, table.output % table.p, output)
    //         }
    //     }
    // }

    // func TestMultinomial2(t *testing.T) {
    //     tables := []struct {
    //         l []int
    //         output int
    //     }{
    //         {[]int {1, 2}, 1},
    //         {[]int {1, 3}, 0},
    //         {[]int {1, 4}, 1},
    //         {[]int {2, 4}, 1},
    //         {[]int {1, 5}, 0},
    //         {[]int {2, 5}, 1},
    //         {[]int {2, 6}, 0},
    //         {[]int {2, 4, 8}, 1},
    //     }
    //     for _, table := range tables {
    //         output := Multinomial2(table.l)
    //         if output != table.output {
    //             t.Errorf("Ran Multinomial2(%v) expected %v, got %v", table.l, table.output, output)
    //         }
    //     }
    // }

    // func TestBinomial2(t *testing.T) {
    //     tables := []struct {
    //         n int
    //         k int
    //         output int
    //     }{
    //         {4, 2, 0},
    //         {72, 46, 0},
    //         {82, 66, 1},
    //         {165, 132, 1},
    //         {169, 140, 0},
    //     }
    //     for _, table := range tables {
    //         output := Binomial2(table.n, table.k)
    //         if output != table.output {
    //             t.Errorf("Ran Binomial2(%v,%v) expected %v, got %v", table.n, table.k, table.output, output)
    //         }
    //     }
    // }

    #[test]
    fn binomial_test() {
        let entries = [
            [2, 2, 1, 0],
            [2, 3, 1, 1],
            [3, 1090, 730, 1],
            [7, 3, 2, 3],
        ];

        for entry in &entries {
            assert_eq!(entry[3] as u32, binomial(entry[0] as u32, entry[1], entry[2]));
        }
    }

    #[test]
    fn binomial_vs_monomial() {
        for &p in &[2, 3, 5, 7, 11] {
            for l in 0 .. 20 {
                for m in 0 .. 20 {
                    assert_eq!(binomial(p, (l + m) as i32, m as i32), multinomial(p, &mut [l, m]))
                }
            }
        }
    }
}

// Mathematica:
// StringReplace[
//  "[\n    " <> # <> "\n]" &[
//   StringJoin @@
//    StringReplace[
//     ToString /@
//      Riffle[Function[p,
//         PadRight[
//          PadRight[#, Prime[8]] & /@
//           Table[Mod[Binomial[n, k], p], {n, 0, p - 1}, {k, 0, p - 1}],
//           Prime[8], {Table[0, {Prime[8]}]}]] /@ Prime[Range[8]],
//       ",\n    "], {"{" -> "[", "}" -> "]"}]], "], " -> "],\n     "]
static BINOMIAL_TABLE : [[[u32; 19]; 19]; 8] = [
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 5, 3, 3, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 6, 1, 6, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 6, 4, 9, 4, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 7, 10, 2, 2, 10, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 8, 6, 1, 4, 1, 6, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 9, 3, 7, 5, 5, 7, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 6, 2, 7, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 7, 8, 9, 9, 8, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 8, 2, 4, 5, 4, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 9, 10, 6, 9, 9, 6, 10, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 10, 6, 3, 2, 5, 2, 3, 6, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 11, 3, 9, 5, 7, 7, 5, 9, 3, 11, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 12, 1, 12, 1, 12, 1, 12, 1, 12, 1, 12, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 6, 15, 3, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 7, 4, 1, 1, 4, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 8, 11, 5, 2, 5, 11, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 9, 2, 16, 7, 7, 16, 2, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 10, 11, 1, 6, 14, 6, 1, 11, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 11, 4, 12, 7, 3, 3, 7, 12, 4, 11, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 12, 15, 16, 2, 10, 6, 10, 2, 16, 15, 12, 1, 0, 0, 0, 0, 0, 0],
     [1, 13, 10, 14, 1, 12, 16, 16, 12, 1, 14, 10, 13, 1, 0, 0, 0, 0, 0],
     [1, 14, 6, 7, 15, 13, 11, 15, 11, 13, 15, 7, 6, 14, 1, 0, 0, 0, 0],
     [1, 15, 3, 13, 5, 11, 7, 9, 9, 7, 11, 5, 13, 3, 15, 1, 0, 0, 0],
     [1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 16, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 6, 15, 1, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 7, 2, 16, 16, 2, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 8, 9, 18, 13, 18, 9, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 9, 17, 8, 12, 12, 8, 17, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 10, 7, 6, 1, 5, 1, 6, 7, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 11, 17, 13, 7, 6, 6, 7, 13, 17, 11, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 12, 9, 11, 1, 13, 12, 13, 1, 11, 9, 12, 1, 0, 0, 0, 0, 0, 0],
     [1, 13, 2, 1, 12, 14, 6, 6, 14, 12, 1, 2, 13, 1, 0, 0, 0, 0, 0],
     [1, 14, 15, 3, 13, 7, 1, 12, 1, 7, 13, 3, 15, 14, 1, 0, 0, 0, 0],
     [1, 15, 10, 18, 16, 1, 8, 13, 13, 8, 1, 16, 18, 10, 15, 1, 0, 0, 0],
     [1, 16, 6, 9, 15, 17, 9, 2, 7, 2, 9, 17, 15, 9, 6, 16, 1, 0, 0],
     [1, 17, 3, 15, 5, 13, 7, 11, 9, 9, 11, 7, 13, 5, 15, 3, 17, 1, 0],
     [1, 18, 1, 18, 1, 18, 1, 18, 1, 18, 1, 18, 1, 18, 1, 18, 1, 18, 1]]
];
