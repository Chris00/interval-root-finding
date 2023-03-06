//! Interval root finding algorithms
//!

use std::{collections::{VecDeque, BinaryHeap}};

/// Requirements for an interval type to be usable for the root
/// finding algorithms.
pub trait Interval: Copy + Eq {
    /// Return `true` if `self` is empty.
    fn is_empty(self) -> bool;

    /// Return `true` if `self` contains 0.
    fn contains_zero(self) -> bool;

    /// Return `true` if `self` is a subset of `rhs`.
    fn subset(self, rhs: Self) -> bool;

    /// Width of the interval `self`.
    fn wid(self) -> f64;

    /// Magnitude of the interval `self`.
    fn mag(self) -> f64;

    /// Return a couple of intervals `(x, y)` such that `x` ∪ `y` ⊇ `self`.
    /// Ideally, `self` should be bisected around the middle point.
    /// None of the two intervals should be empty.
    fn bisect(self) -> (Self, Self);

    /// Return the intervals `[l, m, r]` where `m` is
    /// (`other` + [-ϵ, ϵ]) ∩ `self`, where ϵ = `inflate`, and `l` and
    /// `r` are the (possibly empty) components of `self` ∖ `m`.
    fn trisect(self, other: Self, inflate: f64) -> [Self; 3];

    /// Return a point inside the interval `self`.  It should return
    /// singleton interval (an interval containing only one value).
    fn midpoint(self) -> Self;

    /// Return `self` - `rhs`.
    fn sub(self, rhs: Self) -> Self;

    /// Return `self` / `rhs` assuming 0 ∉ `rhs` (as checked by
    /// [`Self::contains_zero`]).
    fn div(self, rhs: Self) -> Self;

    /// Two-output division: `numerator` / `self`.
    fn mul_rev_to_pair(self, numerator: Self) -> [Self; 2];

    /// Return the intersection of `self` and `other`.
    fn intersection(self, rhs: Self) -> Self;
}

#[cfg(feature = "inari")]
impl Interval for inari::Interval {
    #[inline]
    fn is_empty(self) -> bool { self.is_empty() }

    #[inline]
    fn contains_zero(self) -> bool { self.contains(0.) }

    #[inline]
    fn subset(self, rhs: Self) -> bool { self.subset(rhs) }

    #[inline]
    fn wid(self) -> f64 { self.wid() }

    #[inline]
    fn mag(self) -> f64 { self.mag() }

    #[inline]
    fn bisect(self) -> (Self, Self) {
        use inari::interval as i;
        let m = self.mid();
        (i!(self.inf(), m).unwrap(), i!(m, self.sup()).unwrap())
    }

    #[inline]
    fn trisect(self, other: Self, inflate: f64) -> [Self; 3] {
        use inari::{Interval as I, interval as i};
        let m_inf = other.inf() - inflate;
        let m_sup = other.sup() + inflate;
        if m_inf <= self.inf() {
            if m_sup >= self.sup() {
                [I::EMPTY, self, I::EMPTY]
            } else {
                [I::EMPTY,
                 i!(self.inf(), m_sup).unwrap(),
                 i!(m_sup, self.sup()).unwrap()]
            }
        } else if m_sup >= self.sup() {
            [i!(self.inf(), m_inf).unwrap(),
             i!(m_inf, self.sup()).unwrap(),
             I::EMPTY]
        } else { // self.inf() < m_inf ∧ m_sup < self.sup()
            [i!(self.inf(), m_inf).unwrap(),
             i!(m_inf, m_sup).unwrap(),
             i!(m_sup, self.sup()).unwrap()]
        }
    }

    #[inline]
    fn midpoint(self) -> Self {
        let m = self.mid();
        inari::interval!(m, m).unwrap()
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self { self - rhs }

    #[inline]
    fn div(self, rhs: Self) -> Self { self / rhs }

    #[inline]
    fn mul_rev_to_pair(self, numerator: Self) -> [Self; 2] {
        self.mul_rev_to_pair(numerator)
    }

    #[inline]
    fn intersection(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}


/// Existence and uniqueness of the root in a region.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Status {
    /// There is a single root in the interval or interval box.
    Unique,
    /// There is at most one root in the interval or interval box.
    AtMost1,
    /// It is not known how many roots there are in the interval or
    /// interval box.
    Unknown,
}

/// Default relative tolerance.
const RTOL: f64 = 4. * f64::EPSILON;
/// Default absolute tolerance.
const ATOL: f64 = 2e-12;

/// Interval bisection algorithm.
///
/// An interval `x` is accepted if `x.wid() ≤ rtol * x.mag() + atol`
/// where `rtol` and `atol` are the relative and absolute tolerance
/// respectively.  The default `rtol` is `4. * f64::EPSILON` and the
/// defailt `atol` is `2e-12`.  A maximum number of evaluations of `f`
/// may also be set.  Its defailt is `1000` so that the main stopping
/// criteria are the tolerances.
///
/// # Example
///
/// ```
/// use inari::{Interval as I, const_interval as c};
/// use interval_root_finding::bisection1d;
/// let roots = bisection1d(|x: I| x.sqr() - c!(2., 2.), c!(0., 2.)).roots();
/// assert_eq!(roots.len(), 1);
/// assert!(roots[0].contains(2f64.sqrt()));
/// ```
pub fn bisection1d<I, F>(f: F, x: I) -> Bisection<I, F>
where F: FnMut(I) -> I,
      I: Interval {
    Bisection { f, x, maxeval: 1000, rtol: RTOL, atol: ATOL }
}

pub struct Bisection<I, F> {
    f: F,
    x: I,
    maxeval: usize, // maxmimum number of evaluations of `f`
    rtol: f64,
    atol: f64,
}

macro_rules! set_options { ($struct: ident, $($v: ident),*) => {
    impl<$($v,)*> $struct<$($v,)*> {
        /// Set the maximum number of evaluations of the function.
        ///
        /// Setting `n` to `0` is equivalent to set the maximum number
        /// supported by `usize`.
        pub fn max_eval(mut self, n: usize) -> Self {
            if n == 0 {
                self.maxeval = usize::MAX;
            } else {
                self.maxeval = n;
            }
            self
        }

        /// Set the relative tolerance.
        pub fn rtol(mut self, rtol: f64) -> Self {
            if rtol <= 0. {
                panic!("{}::rtol = {:e} ≤ 0", stringify!($struct), rtol);
            }
            self.rtol = rtol;
            self
        }

        /// Set the absolute tolerance.
        pub fn atol(mut self, atol: f64) -> Self {
            if atol < 0. {
                panic!("{}::atol = {:e} < 0", stringify!($struct), atol);
            }
            self.atol = atol;
            self
        }
    }
}}

set_options!(Bisection, I, F);

impl<I, F> Bisection<I, F>
where F: FnMut(I) -> I,
      I: Interval {
    /// Return a collection of nonempty intervals, each possibly
    /// containing a root.
    pub fn roots(&mut self) -> Vec<I> {
        let mut roots = vec![];
        let mut n = self.maxeval;
        debug_assert!(n >= 1);
        // Do a "breadth first" partitioning of intervals.
        let mut queue = VecDeque::new();
        queue.push_back(self.x);
        while n > 0 {
            if let Some(x) = queue.pop_front() {
                let fx = (self.f)(x);
                n -= 1;
                if fx.contains_zero() {
                    // Thus `x` ≠ ∅ (possible for the first interval only).
                    if x.wid() <= self.rtol * x.mag() + self.atol {
                        roots.push(x);
                        continue;
                    }
                    let (x1, x2) = x.bisect();
                    queue.push_back(x1);
                    queue.push_back(x2);
                }
            } else {
                break;
            }
        }
        for x in queue { roots.push(x) }
        roots
    }
}

/// Interval Newton algorithm.
///
/// An interval `x` is accepted if `x.wid() ≤ rtol * x.mag() + atol`
/// where `rtol` and `atol` are the relative and absolute tolerance
/// respectively.  The default `rtol` is `4. * f64::EPSILON` and the
/// defailt `atol` is `2e-12`.  A maximum number of evaluations of `f`
/// may also be set.  Its defailt is `1000` so that the main stopping
/// criteria are the tolerances.
///
/// # Example
///
/// ```
/// use inari::{Interval as I, const_interval as c};
/// use interval_root_finding::{newton1d, Status};
/// const TWO: I = c!(2., 2.);
/// let roots = newton1d(|x| x.sqr() - TWO, |x| TWO * x, c!(0., 2.)).roots();
/// assert_eq!(roots.len(), 1);
/// assert!(roots[0].0.contains(2f64.sqrt()));
/// assert_eq!(roots[0].1, Status::Unique);
/// ```
pub fn newton1d<I, F, DF>(f: F, df: DF, x: I) -> Newton<I, F, DF>
where F: FnMut(I) -> I,
      DF: FnMut(I) -> I,
      I: Interval {
    Newton { f, df, x, maxeval: 1000, rtol: RTOL, atol: ATOL }
}

/// Set options for the interval Newton method.
pub struct Newton<I, F, DF> {
    f: F,
    df: DF,
    x: I,
    maxeval: usize, // maxmimum number of evaluations of `f`
    rtol: f64,
    atol: f64,
}

set_options!(Newton, I, F, DF);

/// Elements of the priority queue.
struct Elem<I> {
    x: I,
    status: Status,
    priority: f64,
}

impl<I> PartialEq for Elem<I> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<I> Eq for Elem<I> {}

impl<I> Ord for Elem<I> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

impl<I> PartialOrd for Elem<I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Push `x` to the priority queue `pq`.
#[inline]
fn pq_push<I>(pq: &mut BinaryHeap<Elem<I>>, x: I, status: Status)
where I: Interval {
    pq.push(Elem { x, status, priority: x.wid() })
}

impl<I, F, DF> Newton<I, F, DF>
where F: FnMut(I) -> I,
      DF: FnMut(I) -> I,
      I: Interval {
    /// Return a collection of nonempty intervals, each possibly
    /// containing a root.
    pub fn roots(&mut self) -> Vec<(I, Status)> {
        let mut roots = vec![];
        let mut n = self.maxeval;
        debug_assert!(n >= 1);
        // Priority queue to reduce the largest intervals first.
        let mut pq = BinaryHeap::new();
        if self.x.is_empty() {
            return roots
        }
        pq_push(&mut pq, self.x, Status::Unknown);
        while n > 0 {
            if let Some(Elem { x, status, .. }) = pq.pop() {
                if x.wid() <= self.rtol * x.mag() + self.atol {
                    roots.push((x, status));
                    continue;
                }
                let m = x.midpoint();
                let fm = (self.f)(m);
                let dfx = (self.df)(x);
                n -= 1;
                if dfx.contains_zero() {
                    if fm.contains_zero() {
                        // fm / dfx = ℝ so the Newton step will not
                        // shrink the interval.  But the midpoint
                        // (supposedly a singleton interval) is
                        // actually a zero up to the precision of `f`!
                        let tol = self.rtol * m.mag() + self.atol;
                        // Trisect so that the interval around `m`
                        // does not pass the tolerance test and an
                        // additional Newton step is performed.
                        for x in x.trisect(m, tol) {
                            if !x.is_empty() {
                                pq_push(&mut pq, x, status)
                            }
                        }
                    } else {
                        for q in dfx.mul_rev_to_pair(fm) {
                            let x = m.sub(q).intersection(x);
                            if !x.is_empty() { // same status, no improvement
                                pq_push(&mut pq, x, status)
                            }
                        }
                    }
                } else { // 0 ∉ dfx
                    let mut x1 = m.sub(fm.div(dfx)); // ≠ ∅
                    let status1;
                    if x1.subset(x) {
                        status1 = Status::Unique;
                    } else {
                        x1 = x1.intersection(x);
                        if x1.is_empty() { continue } // no root here
                        status1 = status.min(Status::AtMost1);
                    }
                    if x1 == x { // Newton step lead no improvement
                        roots.push((x1, status1))
                    } else {
                        pq_push(&mut pq, x1, status1)
                    }
                }
            } else {
                break;
            }
        }
        for e in pq { roots.push((e.x, e.status)) }
        roots
    }
}


#[cfg(test)]
mod tests {
    use inari::const_interval as c;
    use super::*;

    #[test]
    fn test_bisection_empty() {

    }

    #[test]
    fn test_bisection_basic() {
        let r = bisection1d(|x| x.sqr() - c!(2., 2.), c!(-2., 2.)).roots();
        assert_eq!(r.len(), 2);
        assert!(r.iter().any(|x| x.contains(2f64.sqrt())));
    }
}
