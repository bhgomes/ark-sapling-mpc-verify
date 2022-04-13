//! Arkworks Trusted Setup

use ark_ec::{AffineCurve, PairingEngine};
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
use rand::{rngs::OsRng, SeedableRng};
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

    /// Custom Pair
    ///
    /// Unfortunately, we need to do this trick to reduce the number of clones/copies because of
    /// deficiencies in the arkworks pairing APIs.
    type Pair: From<Pair<Self>> + Into<Pair<Self>>;

    /// Output of the Pairing
    type Output: PartialEq;

    /// Evaluates the pairing on `pair`.
    fn eval(pair: &Self::Pair) -> Self::Output;

    /// Checks if `lhs` and `rhs` evaluate to the same point under the pairing function.
    #[inline]
    fn has_same(lhs: &Self::Pair, rhs: &Self::Pair) -> bool {
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
        let lhs = (lhs.0.into(), lhs.1.into()).into();
        let rhs = (rhs.0.into(), rhs.1.into()).into();
        Self::has_same(&lhs, &rhs).then(|| (lhs.into(), rhs.into()))
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
    type Pair = Pair<Self>;
    type Output = P::Fqk;

    #[inline]
    fn eval(pair: &Self::Pair) -> Self::Output {
        P::product_of_pairings(iter::once(pair))
    }
}

/// Powers of Tau Trusted Setup
pub mod powersoftau {
    use super::*;

    /// Curve
    pub trait Group: AddAssign + Clone + PartialEq + Send + Sync + UniformRand + Zero {
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
        G::rand(&mut ChaCha20Rng::from_seed(into_array_unchecked(seed)))
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
    pub enum PairingGroupsDeserializeError<G1, G2> {
        ///
        IsZero,

        ///
        G1Error(G1),

        ///
        G2Error(G2),
    }

    impl<G1, G2> PairingGroupsDeserializeError<G1, G2> {
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
    type DeserializeErrorType<C, M> = PairingGroupsDeserializeError<
        <<C as Configuration>::G1 as Serde<M>>::Error,
        <<C as Configuration>::G2 as Serde<M>>::Error,
    >;

    ///
    pub struct DeserializeError<C, M>(DeserializeErrorType<C, M>)
    where
        C: Configuration,
        C::G1: Serde<M>,
        C::G2: Serde<M>;

    impl<C, M> DeserializeError<C, M>
    where
        C: Configuration,
        C::G1: Serde<M>,
        C::G2: Serde<M>,
    {
        ///
        #[inline]
        fn read_g1<R>(reader: &mut R) -> Result<NonZero<C::G1>, Self>
        where
            R: Read,
        {
            NonZero::deserialize(reader)
                .map_err(DeserializeErrorType::<C, M>::map_g1)
                .map_err(Self)
        }

        ///
        #[inline]
        fn read_g2<R>(reader: &mut R) -> Result<NonZero<C::G2>, Self>
        where
            R: Read,
        {
            NonZero::deserialize(reader)
                .map_err(DeserializeErrorType::<C, M>::map_g2)
                .map_err(Self)
        }
    }

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

    impl<C, M> Serde<M> for PublicKey<C>
    where
        C: Configuration,
        C::G1: Serde<M>,
        C::G2: Serde<M>,
    {
        type Error = DeserializeError<C, M>;

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
                tau_g1_ratio: (Self::Error::read_g1(reader)?, Self::Error::read_g1(reader)?),
                alpha_g1_ratio: (Self::Error::read_g1(reader)?, Self::Error::read_g1(reader)?),
                beta_g1_ratio: (Self::Error::read_g1(reader)?, Self::Error::read_g1(reader)?),
                tau_g2: Self::Error::read_g2(reader)?,
                alpha_g2: Self::Error::read_g2(reader)?,
                beta_g2: Self::Error::read_g2(reader)?,
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
        type Error = DeserializeError<C, M>;

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
            let mut tau_powers_g1 = Vec::with_capacity(C::G1_POWERS);
            for _ in 0..C::G1_POWERS {
                tau_powers_g1.push(Self::Error::read_g1(reader)?.into_inner());
            }
            let mut tau_powers_g2 = Vec::with_capacity(C::G2_POWERS);
            for _ in 0..C::G2_POWERS {
                tau_powers_g2.push(Self::Error::read_g2(reader)?.into_inner());
            }
            let mut alpha_tau_powers_g1 = Vec::with_capacity(C::G2_POWERS);
            for _ in 0..C::G2_POWERS {
                alpha_tau_powers_g1.push(Self::Error::read_g1(reader)?.into_inner());
            }
            let mut beta_tau_powers_g1 = Vec::with_capacity(C::G2_POWERS);
            for _ in 0..C::G2_POWERS {
                beta_tau_powers_g1.push(Self::Error::read_g1(reader)?.into_inner());
            }
            Ok(Self {
                tau_powers_g1,
                tau_powers_g2,
                alpha_tau_powers_g1,
                beta_tau_powers_g1,
                beta_g2: Self::Error::read_g2(reader)?.into_inner(),
            })
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

/// Non-Zero Type Wrapper
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NonZero<T>(T)
where
    T: Zero;

impl<T> NonZero<T>
where
    T: Zero,
{
    /// Builds a new [`NonZero`] for `t` returning `None` if [`Zero::is_zero`] returns `true` on `t`.
    #[inline]
    pub fn new(t: T) -> Option<Self> {
        if t.is_zero() {
            None
        } else {
            Some(Self(t))
        }
    }

    /// Borrows the underlying non-zero value.
    #[inline]
    pub fn get(&self) -> &T {
        &self.0
    }

    /// Returns the non-zero value unwrapping it back into the base type.
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
        self.get()
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

/// Deserialization Error for [`NonZero`]
pub enum NonZeroDeserializeError<E> {
    /// Value returns `true` on [`Zero::is_zero`]
    IsZero,

    /// Plain Value Deserialization Error
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

/// Sapling MPC
pub mod sapling {
    use super::*;
    use ark_bls12_381::Bls12_381 as ArkBls12_381;
    use core::ops::Add;
    use powersoftau::{Configuration, Group};
    use rand::Rng;

    ///
    #[inline]
    fn write_group<G, X, Y, const N: usize>(point: &G, write_x: X, write_y: Y) -> [u8; N]
    where
        G: AffineCurve,
        X: FnOnce(&G, &mut &mut [u8; N]),
        Y: FnOnce(&G, &mut &mut [u8; N]),
    {
        let mut buffer = [0; N];
        if point.is_zero() {
            // Set the second-most significant bit to indicate this point is at infinity.
            buffer[0] |= 1 << 6;
        } else {
            let mut writer = &mut buffer;
            write_x(point, &mut writer);
            write_y(point, &mut writer);
        }
        buffer
    }

    ///
    #[inline]
    fn write_group_compressed<G, X, Y, const N: usize>(
        point: &G,
        write_x: X,
        compare_y: Y,
    ) -> [u8; N]
    where
        G: AffineCurve,
        X: FnOnce(&G, &mut &mut [u8; N]),
        Y: FnOnce(&G) -> bool,
    {
        let mut buffer = [0; N];
        if point.is_zero() {
            // Set the second-most significant bit to indicate this point is at infinity.
            buffer[0] |= 1 << 6;
        } else {
            write_x(point, &mut &mut buffer);

            // Set the third most significant bit if the correct y-coordinate is lexicographically
            // largest.
            if compare_y(point) {
                buffer[0] |= 1 << 5;
            }
        }
        // Set highest bit to distinguish this as a compressed element.
        buffer[0] |= 1 << 7;
        buffer
    }

    ///
    type G1Type = <ArkBls12_381 as PairingEngine>::G1Affine;

    ///
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct G1(G1Type);

    impl Add for G1 {
        type Output = Self;

        #[inline]
        fn add(mut self, rhs: Self) -> Self::Output {
            self.add_assign(rhs);
            self
        }
    }

    impl AddAssign for G1 {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.0.add_assign(&rhs.0)
        }
    }

    impl Group for G1 {
        type Scalar = <G1Type as AffineCurve>::ScalarField;

        #[inline]
        fn mul(&self, scalar: &Self::Scalar) -> Self {
            Self(self.0.mul(*scalar).into())
        }
    }

    // TODO: 96 Bytes
    impl Serde for G1 {
        type Error = ();

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            todo!()
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            todo!()
        }
    }

    // TODO: 48 Bytes
    impl Serde<Compressed> for G1 {
        type Error = ();

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            todo!()
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            todo!()
        }
    }

    impl UniformRand for G1 {
        #[inline]
        fn rand<R>(rng: &mut R) -> Self
        where
            R: Rng + ?Sized,
        {
            todo!()
        }
    }

    impl Zero for G1 {
        #[inline]
        fn zero() -> Self {
            Self(Zero::zero())
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    ///
    type G2Type = <ArkBls12_381 as PairingEngine>::G2Affine;

    ///
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct G2(G2Type);

    impl Add for G2 {
        type Output = Self;

        #[inline]
        fn add(mut self, rhs: Self) -> Self::Output {
            self.add_assign(rhs);
            self
        }
    }

    impl AddAssign for G2 {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.0.add_assign(&rhs.0)
        }
    }

    impl Group for G2 {
        type Scalar = <G2Type as AffineCurve>::ScalarField;

        #[inline]
        fn mul(&self, scalar: &Self::Scalar) -> Self {
            Self(self.0.mul(*scalar).into())
        }
    }

    // TODO: 192 Bytes
    impl Serde for G2 {
        type Error = ();

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            todo!()
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            todo!()
        }
    }

    // TODO: 96 Bytes
    impl Serde<Compressed> for G2 {
        type Error = ();

        #[inline]
        fn serialize<W>(&self, writer: &mut W) -> Result<(), io::Error>
        where
            W: Write,
        {
            todo!()
        }

        #[inline]
        fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
        where
            R: Read,
        {
            todo!()
        }
    }

    impl UniformRand for G2 {
        #[inline]
        fn rand<R>(rng: &mut R) -> Self
        where
            R: Rng + ?Sized,
        {
            todo!()
        }
    }

    impl Zero for G2 {
        #[inline]
        fn zero() -> Self {
            Self(Zero::zero())
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    ///
    type G1PreparedType = <ArkBls12_381 as PairingEngine>::G1Prepared;

    ///
    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    pub struct G1Prepared(G1PreparedType);

    impl From<G1> for G1Prepared {
        #[inline]
        fn from(point: G1) -> Self {
            Self(point.0.into())
        }
    }

    ///
    type G2PreparedType = <ArkBls12_381 as PairingEngine>::G2Prepared;

    ///
    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    pub struct G2Prepared(G2PreparedType);

    impl From<G2> for G2Prepared {
        #[inline]
        fn from(point: G2) -> Self {
            Self(point.0.into())
        }
    }

    ///
    pub struct Pair((G1PreparedType, G2PreparedType));

    impl From<super::Pair<Bls12_381>> for Pair {
        #[inline]
        fn from(pair: super::Pair<Bls12_381>) -> Self {
            Self(((pair.0).0, (pair.1).0))
        }
    }

    impl From<Pair> for super::Pair<Bls12_381> {
        #[inline]
        fn from(pair: Pair) -> Self {
            (G1Prepared(pair.0 .0), G2Prepared(pair.0 .1))
        }
    }

    ///
    pub struct Bls12_381;

    impl Pairing for Bls12_381 {
        type G1 = G1;
        type G2 = G2;
        type G1Prepared = G1Prepared;
        type G2Prepared = G2Prepared;
        type Pair = Pair;
        type Output = <ArkBls12_381 as PairingEngine>::Fqk;

        #[inline]
        fn eval(pair: &Self::Pair) -> Self::Output {
            ArkBls12_381::product_of_pairings(iter::once(&pair.0))
        }
    }

    ///
    pub struct Sapling;

    impl Configuration for Sapling {
        type G1 = G1;
        type G2 = G2;
        type Pairing = Bls12_381;

        const G1_POWERS: usize = (Self::G2_POWERS << 1) - 1;
        const G2_POWERS: usize = 1 << 21;

        #[inline]
        fn g1_prime_subgroup_generator() -> Self::G1 {
            G1(G1Type::prime_subgroup_generator())
        }

        #[inline]
        fn g2_prime_subgroup_generator() -> Self::G2 {
            G2(G2Type::prime_subgroup_generator())
        }
    }
}

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
