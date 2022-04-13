//! Arkworks Sapling Powers of Tau

use ark_ec::PairingEngine;
use ark_ff::{UniformRand, Zero};
use blake2::{Blake2b, Digest};
use byteorder::{BigEndian, ReadBytesExt};
use core::{
    fmt::{self, Debug},
    hash::Hash,
    iter,
    marker::PhantomData,
    ops::{AddAssign, Deref},
};
use rand::{rngs::OsRng, CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::io::{self, Read, Write};

#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Trusted Setup Verifier
pub trait TrustedSetup {
    /// Accumulator
    type Accumulator;

    /// Contribution Public Key
    type PublicKey;

    /// Challenge
    type Challenge;

    /// Response
    type Response;

    /// Error
    type Error;

    /// Computes the challenge associated to `last` and `last_response` for the next player.
    fn challenge(last: &Self::Accumulator, last_response: &Self::Response) -> Self::Challenge;

    /// Computes the response from `next` and `next_key` to the `challenge` presented by the
    /// previous state.
    fn response(
        next: &Self::Accumulator,
        next_key: &Self::PublicKey,
        challenge: Self::Challenge,
    ) -> Self::Response;

    /// Verifies the transformation from `last` to `next` using the `next_key` and `next_response`
    /// as evidence for the correct update of the state. This method returns the `next` accumulator
    /// and `next_response`.
    fn verify(
        last: Self::Accumulator,
        next: Self::Accumulator,
        next_key: Self::PublicKey,
        next_response: Self::Response,
    ) -> Result<(Self::Accumulator, Self::Response), Self::Error>;

    /// Verifies all accumulator contributions in `iter` chaining from `last` and `last_response`
    /// returning the newest [`Accumulator`](Self::Accumulator) and [`Response`](Self::Response) if
    /// all the contributions in the chain had valid transitions.
    #[inline]
    fn verify_all<E, I>(
        mut last: Self::Accumulator,
        mut last_response: Self::Response,
        iter: I,
    ) -> Result<(Self::Accumulator, Self::Response), Self::Error>
    where
        E: Into<Self::Error>,
        I: IntoIterator<Item = Result<(Self::Accumulator, Self::PublicKey), E>>,
    {
        for item in iter {
            let (next, next_key) = item.map_err(Into::into)?;
            let next_response =
                Self::response(&next, &next_key, Self::challenge(&last, &last_response));
            (last, last_response) = Self::verify(last, next, next_key, next_response)?;
        }
        Ok((last, last_response))
    }
}

/// Trusted Setup Verification State Machine
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "S::Accumulator: Clone, S::Response: Clone"),
    Copy(bound = "S::Accumulator: Copy, S::Response: Copy"),
    Debug(bound = "S::Accumulator: Debug, S::Response: Debug"),
    Default(bound = "S::Accumulator: Default, S::Response: Default"),
    Eq(bound = "S::Accumulator: Eq, S::Response: Eq"),
    Hash(bound = "S::Accumulator: Hash, S::Response: Hash"),
    PartialEq(bound = "S::Accumulator: PartialEq, S::Response: PartialEq")
)]
pub struct State<S>
where
    S: TrustedSetup,
{
    /// Last Accumulator
    last: S::Accumulator,

    /// Last Response
    last_response: S::Response,
}

impl<S> State<S>
where
    S: TrustedSetup,
{
    /// Builds a new [`State`] machine from `last` and `last_response`.
    #[inline]
    pub fn new(last: S::Accumulator, last_response: S::Response) -> Self {
        Self {
            last,
            last_response,
        }
    }

    /// Verifies the update of the accumulator to `next` using the `next_key`. This method returns
    /// the next state with `next` as the accumulator if the verification passed.
    #[inline]
    pub fn verify(self, next: S::Accumulator, next_key: S::PublicKey) -> Result<Self, S::Error> {
        self.verify_all::<S::Error, _>(iter::once(Ok((next, next_key))))
    }

    /// Verifies all accumulator contributions in `iter` chaining from `self` returning the newest
    /// [`State`] if all the contributions in the chain had valid transitions.
    #[inline]
    pub fn verify_all<E, I>(self, iter: I) -> Result<Self, S::Error>
    where
        E: Into<S::Error>,
        I: IntoIterator<Item = Result<(S::Accumulator, S::PublicKey), E>>,
    {
        let (last, last_response) = S::verify_all(self.last, self.last_response, iter)?;
        Ok(Self::new(last, last_response))
    }

    /// Extracts the last accumulator and last response from the state.
    #[inline]
    pub fn into_inner(self) -> (S::Accumulator, S::Response) {
        (self.last, self.last_response)
    }
}

/// Pair from a [`Pairing`]
pub type Pair<P> = (<P as Pairing>::G1Prepared, <P as Pairing>::G2Prepared);

/// Pairing
pub trait Pairing {
    /// Left Group of the Pairing
    type G1;

    /// Right Group of the Pairing
    type G2;

    /// Optimized Pre-computed Form of a [`G1`](Self::G1) Element
    type G1Prepared: From<Self::G1>;

    /// Optimized Pre-computed Form of a [`G2`](Self::G2) Element
    type G2Prepared: From<Self::G2>;

    /// Output of the Pairing
    type Output: PartialEq;

    /// Evaluates the pairing on `pair`.
    fn eval(pair: &Pair<Self>) -> Self::Output;

    /// Checks if `lhs` and `rhs` evaluate to the same point under the pairing function.
    #[inline]
    fn has_same(lhs: &Pair<Self>, rhs: &Pair<Self>) -> bool {
        Self::eval(lhs) == Self::eval(rhs)
    }

    /// Checks if `lhs` and `rhs` evaluate to the same point under the pairing function, returning
    /// `Some` with prepared points if the pairing outcome is the same.
    #[inline]
    fn same<L1, L2, R1, R2>(lhs: (L1, L2), rhs: (R1, R2)) -> Option<(Pair<Self>, Pair<Self>)>
    where
        L1: Into<Self::G1Prepared>,
        L2: Into<Self::G2Prepared>,
        R1: Into<Self::G1Prepared>,
        R2: Into<Self::G2Prepared>,
    {
        let lhs = (lhs.0.into(), lhs.1.into());
        let rhs = (rhs.0.into(), rhs.1.into());
        Self::has_same(&lhs, &rhs).then(|| (lhs, rhs))
    }
}

/// [`Pairing`] Adapter for the Arkworks [`PairingEngine`] Trait
#[derive(derivative::Derivative)]
#[derivative(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArkPairing<P>(PhantomData<P>)
where
    P: PairingEngine;

impl<P> Pairing for ArkPairing<P>
where
    P: PairingEngine,
{
    type G1 = P::G1Affine;
    type G2 = P::G2Affine;
    type G1Prepared = P::G1Prepared;
    type G2Prepared = P::G2Prepared;
    type Output = P::Fqk;

    #[inline]
    fn eval(pair: &Pair<Self>) -> Self::Output {
        P::product_of_pairings(iter::once(pair))
    }
}

/// Powers of Tau Trusted Setup
pub mod powersoftau {
    use super::*;

    /// Curve
    pub trait Group: AddAssign + Clone + PartialEq + Send + Sync + Zero {
        /// Scalar Field
        type Scalar: Send + UniformRand;

        /// Multiplies `self` by `scalar`.
        fn mul(&self, scalar: &Self::Scalar) -> Self;
    }

    ///
    #[inline]
    fn merge_pairs<G>(lhs: &[G], rhs: &[G]) -> (G, G)
    where
        G: Group,
    {
        assert_eq!(lhs.len(), rhs.len());
        into_par_iter(0..lhs.len())
            .map(|_| {
                let mut rng = OsRng;
                G::Scalar::rand(&mut rng)
            })
            .zip(lhs)
            .zip(rhs)
            .map(|((rho, lhs), rhs)| (lhs.mul(&rho), rhs.mul(&rho)))
            .reduce(
                || (Zero::zero(), Zero::zero()),
                |mut acc, next| {
                    acc.0 += next.0;
                    acc.1 += next.1;
                    acc
                },
            )
    }

    ///
    #[inline]
    fn power_pairs<G>(points: &[G]) -> (G, G)
    where
        G: Group,
    {
        merge_pairs(&points[..(points.len() - 1)], &points[1..])
    }

    ///
    #[inline]
    fn hash_to_group<G>(digest: [u8; 32]) -> G
    where
        G: Group,
    {
        let mut digest = digest.as_slice();
        let mut seed = Vec::with_capacity(8);
        for _ in 0..8 {
            let word = digest
                .read_u32::<BigEndian>()
                .expect("This is always possible since we have enough bytes to begin with.");
            seed.extend(word.to_le_bytes());
        }
        sample_group(&mut ChaCha20Rng::from_seed(into_array_unchecked(seed)))
    }

    ///
    #[inline]
    fn sample_group<G, R>(rng: &mut R) -> G
    where
        G: Group,
        R: CryptoRng + RngCore + ?Sized,
    {
        todo!()
    }

    /// Powers of Tau Configuration
    pub trait Configuration {
        /// Left Group of the Pairing
        type G1: Group + Serde + Serde<Compressed>;

        /// Right Group of the Pairing
        type G2: Group + Serde + Serde<Compressed>;

        /// Pairing
        type Pairing: Pairing<G1 = Self::G1, G2 = Self::G2>;

        /// Number of Powers of Tau in G1 Supported by this Setup
        const G1_POWERS: usize;

        /// Number of Powers of Tau in G2 Supported by this Setup
        const G2_POWERS: usize;

        /// Returns a chosen prime subgroup generator for G1.
        fn g1_prime_subgroup_generator() -> Self::G1;

        /// Returns a chosen prime subgroup generator for G2.
        fn g2_prime_subgroup_generator() -> Self::G2;
    }

    ///
    const HASHER_WRITER_EXPECT_MESSAGE: &str =
        "The `Blake2b` hasher's `Write` implementation never returns an error.";

    /// Verification Error
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub enum VerificationError {
        /// Invalid Proof of Knowledge for τ
        TauKnowledgeProof,

        /// Invalid Proof of Knowledge for α
        AlphaKnowledgeProof,

        /// Invalid Proof of Knowledge for β
        BetaKnowledgeProof,

        /// Element Differs from Prime Subgroup Generator in G1
        PrimeSubgroupGeneratorG1,

        /// Element Differs from Prime Subgroup Generator in G2
        PrimeSubgroupGeneratorG2,

        /// Invalid Multiplication of τ
        TauMultiplication,

        /// Invalid Multiplication of α
        AlphaMultiplication,

        /// Invalid Multiplication of β
        BetaMultiplication,

        /// Invalid Computation of Powers of τ
        PowersOfTau,
    }

    ///
    pub enum Error {
        ///
        Io(io::Error),

        ///
        Verification(VerificationError),
    }

    from_variant_impl!(Error, Io, io::Error);
    from_variant_impl!(Error, Verification, VerificationError);

    /// Challenge Digest
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct ChallengeDigest([u8; 64]);

    /// Response Digest
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct ResponseDigest([u8; 64]);

    impl ResponseDigest {
        /// Hashes `self` and the G<sub>1</sub> `ratio` to produce a G<sub>2</sub> group element.
        #[inline]
        pub fn hash_to_group<C>(
            self,
            personalization: u8,
            ratio: &(NonZero<C::G1>, NonZero<C::G1>),
        ) -> C::G2
        where
            C: Configuration,
        {
            let mut hasher = Blake2b::default();
            hasher.update(&[personalization]);
            hasher.update(&self.0);
            Serde::<()>::serialize(&ratio.0, &mut hasher).expect(HASHER_WRITER_EXPECT_MESSAGE);
            Serde::<()>::serialize(&ratio.1, &mut hasher).expect(HASHER_WRITER_EXPECT_MESSAGE);
            hash_to_group(into_array_unchecked(hasher.finalize()))
        }
    }

    impl Default for ResponseDigest {
        #[inline]
        fn default() -> Self {
            Self(into_array_unchecked(Blake2b::default().finalize()))
        }
    }

    impl fmt::Display for ResponseDigest {
        #[inline]
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(&hex::encode(self.0), f)
        }
    }

    ///
    pub enum PublicKeyDeserializeError<G1, G2> {
        ///
        IsZero,

        ///
        G1Error(G1),

        ///
        G2Error(G2),
    }

    impl<G1, G2> PublicKeyDeserializeError<G1, G2> {
        ///
        #[inline]
        fn map_g1(err: NonZeroDeserializeError<G1>) -> Self {
            match err {
                NonZeroDeserializeError::IsZero => Self::IsZero,
                NonZeroDeserializeError::Error(err) => Self::G1Error(err),
            }
        }

        ///
        #[inline]
        fn map_g2(err: NonZeroDeserializeError<G2>) -> Self {
            match err {
                NonZeroDeserializeError::IsZero => Self::IsZero,
                NonZeroDeserializeError::Error(err) => Self::G2Error(err),
            }
        }
    }

    ///
    type PublicKeyDeserializeErrorType<C, M> = PublicKeyDeserializeError<
        <<C as Configuration>::G1 as Serde<M>>::Error,
        <<C as Configuration>::G2 as Serde<M>>::Error,
    >;

    /// Contribution Public Key
    pub struct PublicKey<C>
    where
        C: Configuration,
    {
        ///
        tau_g1_ratio: (NonZero<C::G1>, NonZero<C::G1>),

        ///
        alpha_g1_ratio: (NonZero<C::G1>, NonZero<C::G1>),

        ///
        beta_g1_ratio: (NonZero<C::G1>, NonZero<C::G1>),

        ///
        tau_g2: NonZero<C::G2>,

        ///
        alpha_g2: NonZero<C::G2>,

        ///
        beta_g2: NonZero<C::G2>,
    }

    impl<C> PublicKey<C>
    where
        C: Configuration,
    {
        ///
        #[inline]
        fn read_g1<M, R>(
            reader: &mut R,
        ) -> Result<NonZero<C::G1>, PublicKeyDeserializeErrorType<C, M>>
        where
            C::G1: Serde<M>,
            C::G2: Serde<M>,
            R: Read,
        {
            NonZero::deserialize(reader).map_err(PublicKeyDeserializeError::map_g1)
        }

        ///
        #[inline]
        fn read_g2<M, R>(
            reader: &mut R,
        ) -> Result<NonZero<C::G2>, PublicKeyDeserializeErrorType<C, M>>
        where
            C::G1: Serde<M>,
            C::G2: Serde<M>,
            R: Read,
        {
            NonZero::deserialize(reader).map_err(PublicKeyDeserializeError::map_g2)
        }
    }

    impl<C, M> Serde<M> for PublicKey<C>
    where
        C: Configuration,
        C::G1: Serde<M>,
        C::G2: Serde<M>,
    {
        type Error = PublicKeyDeserializeErrorType<C, M>;

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            self.tau_g1_ratio.0.serialize(writer)?;
            self.tau_g1_ratio.1.serialize(writer)?;
            self.alpha_g1_ratio.0.serialize(writer)?;
            self.alpha_g1_ratio.1.serialize(writer)?;
            self.beta_g1_ratio.0.serialize(writer)?;
            self.beta_g1_ratio.1.serialize(writer)?;
            self.tau_g2.serialize(writer)?;
            self.alpha_g2.serialize(writer)?;
            self.beta_g2.serialize(writer)?;
            Ok(())
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            Ok(Self {
                tau_g1_ratio: (Self::read_g1(reader)?, Self::read_g1(reader)?),
                alpha_g1_ratio: (Self::read_g1(reader)?, Self::read_g1(reader)?),
                beta_g1_ratio: (Self::read_g1(reader)?, Self::read_g1(reader)?),
                tau_g2: Self::read_g2(reader)?,
                alpha_g2: Self::read_g2(reader)?,
                beta_g2: Self::read_g2(reader)?,
            })
        }
    }

    /// Contribution Accumulator
    pub struct Accumulator<C>
    where
        C: Configuration,
    {
        ///
        tau_powers_g1: Vec<C::G1>,

        ///
        tau_powers_g2: Vec<C::G2>,

        ///
        alpha_tau_powers_g1: Vec<C::G1>,

        ///
        beta_tau_powers_g1: Vec<C::G1>,

        ///
        beta_g2: C::G2,
    }

    impl<C> Default for Accumulator<C>
    where
        C: Configuration,
    {
        #[inline]
        fn default() -> Self {
            Self {
                tau_powers_g1: vec![C::g1_prime_subgroup_generator(); C::G1_POWERS],
                tau_powers_g2: vec![C::g2_prime_subgroup_generator(); C::G2_POWERS],
                alpha_tau_powers_g1: vec![C::g1_prime_subgroup_generator(); C::G2_POWERS],
                beta_tau_powers_g1: vec![C::g1_prime_subgroup_generator(); C::G2_POWERS],
                beta_g2: C::g2_prime_subgroup_generator(),
            }
        }
    }

    impl<C, M> Serde<M> for Accumulator<C>
    where
        C: Configuration,
        C::G1: Serde<M>,
        C::G2: Serde<M>,
    {
        type Error = ();

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            for elem in &self.tau_powers_g1 {
                elem.serialize(writer)?;
            }
            for elem in &self.tau_powers_g2 {
                elem.serialize(writer)?;
            }
            for elem in &self.alpha_tau_powers_g1 {
                elem.serialize(writer)?;
            }
            for elem in &self.beta_tau_powers_g1 {
                elem.serialize(writer)?;
            }
            self.beta_g2.serialize(writer)?;
            Ok(())
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            todo!()
        }
    }

    /// Powers of Tau Trusted Setup
    pub struct PowersOfTau<C>(PhantomData<C>)
    where
        C: Configuration;

    impl<C> TrustedSetup for PowersOfTau<C>
    where
        C: Configuration,
    {
        type Accumulator = Accumulator<C>;
        type PublicKey = PublicKey<C>;
        type Challenge = ChallengeDigest;
        type Response = ResponseDigest;
        type Error = Error;

        #[inline]
        fn challenge(last: &Self::Accumulator, last_response: &Self::Response) -> Self::Challenge {
            let mut hasher = Blake2b::default();
            hasher.update(last_response.0);
            Serde::<()>::serialize(last, &mut hasher).expect(HASHER_WRITER_EXPECT_MESSAGE);
            ChallengeDigest(into_array_unchecked(hasher.finalize()))
        }

        #[inline]
        fn response(
            next: &Self::Accumulator,
            next_key: &Self::PublicKey,
            challenge: Self::Challenge,
        ) -> Self::Response {
            let mut hasher = Blake2b::default();
            hasher.update(challenge.0);
            Serde::<Compressed>::serialize(next, &mut hasher).expect(HASHER_WRITER_EXPECT_MESSAGE);
            Serde::<()>::serialize(next_key, &mut hasher).expect(HASHER_WRITER_EXPECT_MESSAGE);
            ResponseDigest(into_array_unchecked(hasher.finalize()))
        }

        #[inline]
        fn verify(
            last: Self::Accumulator,
            next: Self::Accumulator,
            next_key: Self::PublicKey,
            next_response: Self::Response,
        ) -> Result<(Self::Accumulator, Self::Response), Self::Error> {
            let tau_g2_s = next_response.hash_to_group::<C>(0, &next_key.tau_g1_ratio);
            let alpha_g2_s = next_response.hash_to_group::<C>(1, &next_key.alpha_g1_ratio);
            let beta_g2_s = next_response.hash_to_group::<C>(2, &next_key.beta_g1_ratio);

            let ((_, key_tau_g2), (_, tau_g2_s)) = C::Pairing::same(
                (
                    next_key.tau_g1_ratio.0.into_inner(),
                    next_key.tau_g2.into_inner(),
                ),
                (next_key.tau_g1_ratio.1.into_inner(), tau_g2_s),
            )
            .ok_or(VerificationError::TauKnowledgeProof)?;
            let ((_, key_alpha_g2), (_, alpha_g2_s)) = C::Pairing::same(
                (
                    next_key.alpha_g1_ratio.0.into_inner(),
                    next_key.alpha_g2.into_inner(),
                ),
                (next_key.alpha_g1_ratio.1.into_inner(), alpha_g2_s),
            )
            .ok_or(VerificationError::AlphaKnowledgeProof)?;
            let ((_, key_beta_g2), (_, beta_g2_s)) = C::Pairing::same(
                (
                    next_key.beta_g1_ratio.0.into_inner(),
                    next_key.beta_g2.into_inner(),
                ),
                (next_key.beta_g1_ratio.1.into_inner(), beta_g2_s),
            )
            .ok_or(VerificationError::BetaKnowledgeProof)?;

            if next.tau_powers_g1[0] != C::g1_prime_subgroup_generator() {
                return Err(VerificationError::PrimeSubgroupGeneratorG1.into());
            }
            if next.tau_powers_g2[0] != C::g2_prime_subgroup_generator() {
                return Err(VerificationError::PrimeSubgroupGeneratorG2.into());
            }

            C::Pairing::same(
                (last.tau_powers_g1[1].clone(), key_tau_g2),
                (next.tau_powers_g1[1].clone(), tau_g2_s),
            )
            .ok_or(VerificationError::TauMultiplication)?;
            C::Pairing::same(
                (last.alpha_tau_powers_g1[0].clone(), key_alpha_g2),
                (next.alpha_tau_powers_g1[0].clone(), alpha_g2_s),
            )
            .ok_or(VerificationError::AlphaMultiplication)?;
            let ((last_beta_tau_powers_g1_0, _), (next_beta_tau_powers_g1_0, _)) =
                C::Pairing::same(
                    (last.beta_tau_powers_g1[0].clone(), key_beta_g2),
                    (next.beta_tau_powers_g1[0].clone(), beta_g2_s),
                )
                .ok_or(VerificationError::BetaMultiplication)?;
            C::Pairing::same(
                (last_beta_tau_powers_g1_0, next.beta_g2.clone()),
                (next_beta_tau_powers_g1_0, last.beta_g2),
            )
            .ok_or(VerificationError::BetaMultiplication)?;

            let (lhs, rhs) = power_pairs(&next.tau_powers_g2);
            C::Pairing::same(
                (next.tau_powers_g1[0].clone(), rhs),
                (next.tau_powers_g1[1].clone(), lhs),
            )
            .ok_or(VerificationError::PowersOfTau)?;
            let (lhs, rhs) = power_pairs(&next.tau_powers_g1);
            let ((_, next_tau_powers_g2_1), (_, next_tau_powers_g2_0)) = C::Pairing::same(
                (lhs, next.tau_powers_g2[1].clone()),
                (rhs, next.tau_powers_g2[0].clone()),
            )
            .ok_or(VerificationError::PowersOfTau)?;
            let (lhs, rhs) = power_pairs(&next.alpha_tau_powers_g1);
            let ((_, next_tau_powers_g2_1), (_, next_tau_powers_g2_0)) =
                C::Pairing::same((lhs, next_tau_powers_g2_1), (rhs, next_tau_powers_g2_0))
                    .ok_or(VerificationError::PowersOfTau)?;
            let (lhs, rhs) = power_pairs(&next.beta_tau_powers_g1);
            C::Pairing::same((lhs, next_tau_powers_g2_1), (rhs, next_tau_powers_g2_0))
                .ok_or(VerificationError::PowersOfTau)?;

            Ok((next, next_response))
        }
    }
}

/// Implements [`From`]`<$from>` for an enum `$to`, choosing the `$kind` variant.
#[macro_export]
macro_rules! from_variant_impl {
    ($to:ty, $kind:ident, $from:ty) => {
        impl From<$from> for $to {
            #[inline]
            fn from(t: $from) -> Self {
                Self::$kind(t)
            }
        }
    };
}

/// Error Message for the [`into_array_unchecked`] and [`into_boxed_array_unchecked`] messages.
const INTO_UNCHECKED_ERROR_MESSAGE: &str =
    "Input did not have the correct length to match the output array of length";

/// Performs the [`TryInto`] conversion into an array without checking if the conversion succeeded.
#[inline]
pub fn into_array_unchecked<T, V, const N: usize>(value: V) -> [T; N]
where
    V: TryInto<[T; N]>,
{
    match value.try_into() {
        Ok(array) => array,
        _ => unreachable!("{} {:?}.", INTO_UNCHECKED_ERROR_MESSAGE, N),
    }
}

/// Performs the [`TryInto`] conversion into a boxed array without checking if the conversion
/// succeeded.
#[inline]
pub fn into_boxed_array_unchecked<T, V, const N: usize>(value: V) -> Box<[T; N]>
where
    V: TryInto<Box<[T; N]>>,
{
    match value.try_into() {
        Ok(boxed_array) => boxed_array,
        _ => unreachable!("{} {:?}.", INTO_UNCHECKED_ERROR_MESSAGE, N),
    }
}

/// Serialization and Deserialization Trait
///
/// This trait comes with a "mode" flag `M` which specifies an encoding convention.
pub trait Serde<M = ()>: Sized {
    /// Deserialization Error Type
    type Error;

    /// Serializes `self` into the `writer`.
    fn serialize<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write;

    /// Deserializes a value of type `Self` from the `reader`.
    fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read;
}

/// Compressed Serialization and Deserialization Mode
///
/// This mode is used by [`Serde`] to specify a canonical compressed encoding scheme.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Compressed;

///
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NonZero<T>(T)
where
    T: Zero;

impl<T> NonZero<T>
where
    T: Zero,
{
    ///
    #[inline]
    pub fn new(t: T) -> Option<Self> {
        if t.is_zero() {
            None
        } else {
            Some(Self(t))
        }
    }

    ///
    #[inline]
    pub fn get(&self) -> &T {
        &self.0
    }

    ///
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Deref for NonZero<T>
where
    T: Zero,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, M> Serde<M> for NonZero<T>
where
    T: Serde<M> + Zero,
{
    type Error = NonZeroDeserializeError<T::Error>;

    #[inline]
    fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
    where
        W: Write,
    {
        self.0.serialize(writer)
    }

    #[inline]
    fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read,
    {
        Self::new(T::deserialize(reader).map_err(NonZeroDeserializeError::Error)?)
            .ok_or(NonZeroDeserializeError::IsZero)
    }
}

///
pub enum NonZeroDeserializeError<E> {
    ///
    IsZero,

    ///
    Error(E),
}

///
#[cfg(not(feature = "rayon"))]
fn into_par_iter<I>(iter: I) -> I {
    iter
}

///
#[cfg(feature = "rayon")]
fn into_par_iter<I>(iter: I) -> I::Iter
where
    I: IntoParallelIterator,
{
    iter.into_par_iter()
}

/*
/// The accumulator supports circuits with 2^21 multiplication gates.
pub const TAU_POWERS_LENGTH: usize = 1 << 21;

/// More tau powers are needed in G1 because the Groth16 H query
/// includes terms of the form tau^i * (tau^m - 1) = tau^(i+m) - tau^i
/// where the largest i = m - 2, requiring the computation of tau^(2m - 2)
/// and thus giving us a vector length of 2^22 - 1.
pub const TAU_POWERS_G1_LENGTH: usize = (TAU_POWERS_LENGTH << 1) - 1;

///
#[inline]
fn group_uncompressed<G>(point: &G) -> Vec<u8>
where
    G: AffineCurve,
{
    /* TODO: For BLS12-381
    let mut res: [u8; 96] = [0; 96];
    if point.is_zero() {
        // Set the second-most significant bit to indicate this point is at infinity.
        res.0[0] |= 1 << 6;
    } else {
        let mut writer = &mut res.0[..];
        point.x.into_repr().write_be(&mut writer).unwrap();
        point.y.into_repr().write_be(&mut writer).unwrap();
    }
    res
    */
    todo!()
}

///
#[inline]
fn group_compressed<G>(point: &G) -> Vec<u8>
where
    G: AffineCurve,
{
    /* TODO: For BLS12-381
    let mut res: [u8; 48] = [0; 48];
    if point.is_zero() {
        // Set the second-most significant bit to indicate this point is at infinity.
        res.0[0] |= 1 << 6;
    } else {
        {
            let mut writer = &mut res.0[..];
            point.x.into_repr().write_be(&mut writer).unwrap();
        }

        let mut negy = point.y;
        negy.negate();

        // Set the third most significant bit if the correct y-coordinate is lexicographically
        // largest.
        if point.y > negy {
            res.0[0] |= 1 << 5;
        }
    }
    // Set highest bit to distinguish this as a compressed element.
    res.0[0] |= 1 << 7;
    res
    */
    todo!()
}

/*
impl Deserialize for Accumulator<Bls12_381> {
    type Error = io::Error;

    #[inline]
    fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read,
    {
        /*
        pub fn deserialize<R: Read>(
            reader: &mut R,
            compression: UseCompression,
            checked: CheckForCorrectness
        ) -> Result<Self, DeserializationError>
        {
            fn read_all<R: Read, C: CurveAffine>(
                reader: &mut R,
                size: usize,
                compression: UseCompression,
                checked: CheckForCorrectness
            ) -> Result<Vec<C>, DeserializationError>
            {
                fn decompress_all<R: Read, E: EncodedPoint>(
                    reader: &mut R,
                    size: usize,
                ) -> Result<Vec<E::Affine>, DeserializationError>
                {
                    // Read the encoded elements
                    let mut res = vec![E::empty(); size];

                    for encoded in &mut res {
                        reader.read_exact(encoded.as_mut())?;
                    }

                    // Allocate space for the deserialized elements
                    let mut res_affine = vec![E::Affine::zero(); size];

                    let mut chunk_size = res.len() / num_cpus::get();
                    if chunk_size == 0 {
                        chunk_size = 1;
                    }

                    // If any of our threads encounter a deserialization/IO error, catch
                    // it with this.
                    let decoding_error = Arc::new(Mutex::new(None));

                    crossbeam::scope(|scope| {
                        for (source, target) in res.chunks(chunk_size).zip(res_affine.chunks_mut(chunk_size)) {
                            let decoding_error = decoding_error.clone();

                            scope.spawn(move || {
                                for (source, target) in source.iter().zip(target.iter_mut()) {
                                        // Points at infinity are never expected in the accumulator
                                    let res = source.into_affine().map_err(|e| e.into()).and_then(|source| {
                                            if source.is_zero() {
                                                Err(DeserializationError::PointAtInfinity)
                                            } else {
                                                Ok(source)
                                            }
                                        });
                                    match res {
                                        Ok(source) => {
                                            *target = source;
                                        },
                                        Err(e) => {
                                            *decoding_error.lock().unwrap() = Some(e);
                                        }
                                    }
                                }
                            });
                        }
                    });

                    match Arc::try_unwrap(decoding_error).unwrap().into_inner().unwrap() {
                        Some(e) => {
                            Err(e)
                        },
                        None => {
                            Ok(res_affine)
                        }
                    }
                }

                match compression {
                    UseCompression::Yes => decompress_all::<_, C::Compressed>(reader, size, checked),
                    UseCompression::No => decompress_all::<_, C::Uncompressed>(reader, size, checked)
                }
            }
            */
        todo!()
    }
}
*/
*/

///
pub const ROUNDS: usize = 89;

///
fn main() -> io::Result<()> {
    // 1. Load Buffered Reader for Transcript from Disk (or over internet)
    // TODO: ...

    // 2. Verify the accumulators and return the last one.
    // TODO: let accumulator = verify_accumulators(&mut reader)?;

    // 3. Output accumulator powers of tau
    // TODO: let (powers_of_tau_g1, powers_of_tau_g2) = accumulator.collect();

    // 4. Serialize powers of tau to disk
    // TODO: ...

    todo!()
}
