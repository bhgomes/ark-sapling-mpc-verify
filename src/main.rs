//! Arkworks Sapling Powers of Tau

use ark_bls12_381::{Bls12_381, G1Affine, G2Affine};
use ark_ec::{AffineCurve, PairingEngine};
use blake2::{Blake2b, Digest};
use byteorder::{BigEndian, ReadBytesExt};
use core::{fmt, iter};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::io::{self, Read, Write};

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

/// Serialization Trait
pub trait Serialize {
    /// Serializes `self` into `writer`.
    fn serialize<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write;
}

/// Compressed Serialization Trait
pub trait SerializeCompressed {
    /// Serializes a compressed form of `self` into `writer`.
    fn serialize_compressed<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write;
}

/// Deserialization Trait
pub trait Deserialize: Sized {
    ///
    type Error: From<io::Error>;

    ///
    fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read;
}

/// Compressed Deserialization Trait
pub trait DeserializeCompressed: Sized {
    ///
    type Error: From<io::Error>;

    ///
    fn deserialize_compressed<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read;
}

///
const HASHER_WRITER_EXPECT_MESSAGE: &str =
    "The `Blake2b` hasher's `Write` implementation never returns an error.";

/// The accumulator supports circuits with 2^21 multiplication gates.
pub const TAU_POWERS_LENGTH: usize = 1 << 21;

/// More tau powers are needed in G1 because the Groth16 H query
/// includes terms of the form tau^i * (tau^m - 1) = tau^(i+m) - tau^i
/// where the largest i = m - 2, requiring the computation of tau^(2m - 2)
/// and thus giving us a vector length of 2^22 - 1.
pub const TAU_POWERS_G1_LENGTH: usize = (TAU_POWERS_LENGTH << 1) - 1;

///
#[inline]
pub fn hash_to_group<G>(digest: [u8; 32]) -> G
where
    G: AffineCurve,
{
    let mut digest = digest.as_slice();
    let mut seed = Vec::with_capacity(8);
    for _ in 0..8 {
        let word = digest.read_u32::<BigEndian>().expect("");
        seed.extend(word.to_le_bytes());
    }
    sample_group(&mut ChaCha20Rng::from_seed(into_array_unchecked(seed)))
}

///
#[inline]
fn sample_group<G, R>(rng: &mut R) -> G
where
    G: AffineCurve,
    R: CryptoRng + RngCore + ?Sized,
{
    todo!()
}

/*
fn write_point<W, G>(writer: &mut W, p: &G, compression: UseCompression) -> io::Result<()>
where
    W: Write,
    G: CurveAffine,
{
    match compression {
        UseCompression::Yes => writer.write_all(p.into_compressed().as_ref()),
        UseCompression::No => writer.write_all(p.into_uncompressed().as_ref()),
    }
}
*/

///
#[inline]
fn pairing<E>(pair: &(E::G1Prepared, E::G2Prepared)) -> E::Fqk
where
    E: PairingEngine,
{
    E::product_of_pairings(iter::once(pair))
}

///
#[inline]
fn same_pairing<E>(
    lhs: &(E::G1Prepared, E::G2Prepared),
    rhs: &(E::G1Prepared, E::G2Prepared),
) -> bool
where
    E: PairingEngine,
{
    pairing::<E>(lhs) == pairing::<E>(rhs)
}

///
pub enum VerificationError {
    ///
    TauKnowledgeProof,

    ///
    AlphaKnowledgeProof,

    ///
    BetaKnowledgeProof,

    ///
    PrimeSubgroupGeneratorG1,

    ///
    PrimeSubgroupGeneratorG2,
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

///
pub struct PublicKey<E>
where
    E: PairingEngine,
{
    ///
    pub tau_g1_ratio: (E::G1Affine, E::G1Affine),

    ///
    pub alpha_g1_ratio: (E::G1Affine, E::G1Affine),

    ///
    pub beta_g1_ratio: (E::G1Affine, E::G1Affine),

    ///
    pub tau_g2: E::G2Affine,

    ///
    pub alpha_g2: E::G2Affine,

    ///
    pub beta_g2: E::G2Affine,
}

impl Serialize for PublicKey<Bls12_381> {
    #[inline]
    fn serialize<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        writer.write_all(&group_uncompressed(&self.tau_g1_ratio.0))?;
        writer.write_all(&group_uncompressed(&self.tau_g1_ratio.1))?;
        writer.write_all(&group_uncompressed(&self.alpha_g1_ratio.0))?;
        writer.write_all(&group_uncompressed(&self.alpha_g1_ratio.1))?;
        writer.write_all(&group_uncompressed(&self.beta_g1_ratio.0))?;
        writer.write_all(&group_uncompressed(&self.beta_g1_ratio.1))?;
        writer.write_all(&group_uncompressed(&self.tau_g2))?;
        writer.write_all(&group_uncompressed(&self.alpha_g2))?;
        writer.write_all(&group_uncompressed(&self.beta_g2))?;
        Ok(())
    }
}

impl Deserialize for PublicKey<Bls12_381> {
    type Error = io::Error;

    #[inline]
    fn deserialize<R>(reader: &mut R) -> Result<Self, Self::Error>
    where
        R: Read,
    {
        /*
        fn read_uncompressed<C: CurveAffine, R: Read>(reader: &mut R) -> Result<C, DeserializationError> {
            let mut repr = C::Uncompressed::empty();
            reader.read_exact(repr.as_mut())?;
            let v = repr.into_affine()?;

            if v.is_zero() {
                Err(DeserializationError::PointAtInfinity)
            } else {
                Ok(v)
            }
        }

        let tau_g1_s = read_uncompressed(reader)?;
        let tau_g1_s_tau = read_uncompressed(reader)?;

        let alpha_g1_s = read_uncompressed(reader)?;
        let alpha_g1_s_alpha = read_uncompressed(reader)?;

        let beta_g1_s = read_uncompressed(reader)?;
        let beta_g1_s_beta = read_uncompressed(reader)?;

        let tau_g2 = read_uncompressed(reader)?;
        let alpha_g2 = read_uncompressed(reader)?;
        let beta_g2 = read_uncompressed(reader)?;

        Ok(PublicKey {
            tau_g1: (tau_g1_s, tau_g1_s_tau),
            alpha_g1: (alpha_g1_s, alpha_g1_s_alpha),
            beta_g1: (beta_g1_s, beta_g1_s_beta),
            tau_g2: tau_g2,
            alpha_g2: alpha_g2,
            beta_g2: beta_g2
        })
        */
        todo!()
    }
}

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

/// Challenge Digest
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChallengeDigest([u8; 64]);

/// Response Digest
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResponseDigest([u8; 64]);

impl ResponseDigest {
    ///
    #[inline]
    pub fn hash_to_group<E>(
        self,
        personalization: u8,
        ratio: (E::G1Affine, E::G1Affine),
    ) -> E::G2Prepared
    where
        E: PairingEngine,
    {
        let mut hasher = Blake2b::default();
        hasher.update(&[personalization]);
        hasher.update(&self.0);
        hasher.update(group_uncompressed(&ratio.0));
        hasher.update(group_uncompressed(&ratio.1));
        E::G2Prepared::from(hash_to_group(into_array_unchecked(hasher.finalize())))
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
        hex::encode(self.0).fmt(f)
    }
}

///
#[derive(Clone, Eq, PartialEq)]
pub struct Accumulator<E>
where
    E: PairingEngine,
{
    ///
    pub tau_powers_g1: Box<[E::G1Affine; TAU_POWERS_G1_LENGTH]>,

    ///
    pub tau_powers_g2: Box<[E::G2Affine; TAU_POWERS_LENGTH]>,

    ///
    pub alpha_tau_powers_g1: Box<[E::G1Affine; TAU_POWERS_LENGTH]>,

    ///
    pub beta_tau_powers_g1: Box<[E::G1Affine; TAU_POWERS_LENGTH]>,

    ///
    pub beta_g2: E::G2Affine,
}

impl<E> Accumulator<E>
where
    E: PairingEngine,
{
    /// Computes the hash of the challenge file for the player, given the current state of the
    /// accumulator and the last response file hash.
    #[inline]
    pub fn challenge_digest(&self, response: ResponseDigest) -> ChallengeDigest
    where
        Self: Serialize,
    {
        let mut hasher = Blake2b::default();
        hasher.update(response.0);
        self.serialize(&mut hasher)
            .expect(HASHER_WRITER_EXPECT_MESSAGE);
        ChallengeDigest(into_array_unchecked(hasher.finalize()))
    }

    /// Computes the hash of the response file, given the new accumulator, the player's public key,
    /// and the challenge file's hash.
    #[inline]
    pub fn response_digest(&self, key: &PublicKey<E>, challenge: ChallengeDigest) -> ResponseDigest
    where
        Self: SerializeCompressed,
        PublicKey<E>: Serialize,
    {
        let mut hasher = Blake2b::default();
        hasher.update(challenge.0);
        self.serialize_compressed(&mut hasher)
            .expect(HASHER_WRITER_EXPECT_MESSAGE);
        key.serialize(&mut hasher)
            .expect(HASHER_WRITER_EXPECT_MESSAGE);
        ResponseDigest(into_array_unchecked(hasher.finalize()))
    }

    ///
    #[inline]
    pub fn update(
        &mut self,
        next: Accumulator<E>,
        key: PublicKey<E>,
        response_digest: ResponseDigest,
    ) -> Result<(), VerificationError> {
        let tau_g2_s = response_digest.hash_to_group::<E>(0, key.tau_g1_ratio);
        let alpha_g2_s = response_digest.hash_to_group::<E>(1, key.alpha_g1_ratio);
        let beta_g2_s = response_digest.hash_to_group::<E>(2, key.beta_g1_ratio);

        let key_tau_g2 = key.tau_g2.into();
        if !same_pairing::<E>(
            &(key.tau_g1_ratio.0.into(), key_tau_g2),
            &(key.tau_g1_ratio.1.into(), tau_g2_s),
        ) {
            return Err(VerificationError::TauKnowledgeProof);
        }

        let key_alpha_g2 = key.alpha_g2.into();
        if !same_pairing::<E>(
            &(key.alpha_g1_ratio.0.into(), key_alpha_g2),
            &(key.alpha_g1_ratio.0.into(), alpha_g2_s),
        ) {
            return Err(VerificationError::AlphaKnowledgeProof);
        }

        let key_beta_g2 = key.beta_g2.into();
        if !same_pairing::<E>(
            &(key.beta_g1_ratio.0.into(), key_beta_g2),
            &(key.beta_g1_ratio.0.into(), beta_g2_s),
        ) {
            return Err(VerificationError::BetaKnowledgeProof);
        }

        if next.tau_powers_g1[0] != E::G1Affine::prime_subgroup_generator() {
            return Err(VerificationError::PrimeSubgroupGeneratorG1);
        }
        if next.tau_powers_g2[0] != E::G2Affine::prime_subgroup_generator() {
            return Err(VerificationError::PrimeSubgroupGeneratorG2);
        }

        /*
        // Did the participant multiply the previous tau by the new one?
        if !same_ratio(
            (self.tau_powers_g1[1], next.tau_powers_g1[1]),
            (tau_g2_s, key_tau_g2),
        ) {
            return Err(Error::TauMultiplication);
        }

        // Did the participant multiply the previous alpha by the new one?
        if !same_ratio(
            (self.alpha_tau_powers_g1[0], next.alpha_tau_powers_g1[0]),
            (alpha_g2_s, key_alpha_g2),
        ) {
            return Err(Error::AlphaMultiplication);
        }

        // Did the participant multiply the previous beta by the new one?
        if !same_ratio(
            (self.beta_tau_powers_g1[0], next.beta_tau_powers_g1[0]),
            (beta_g2_s, key_beta_g2),
        ) {
            return false;
        }

        if !same_ratio(
            (self.beta_tau_powers_g1[0], next.beta_tau_powers_g1[0]),
            (self.beta_g2, next.beta_g2),
        ) {
            return false;
        }
        */

        /* TODO:
        if !same_ratio(
            (next.tau_powers_g1[0], next.tau_powers_g1[1]),
            power_pairs(&next.tau_powers_g2),
        ) {
            return false;
        }

        let next_tau_powers_g2_0 = E::G2Prepared::from(next.tau_powers_g2[0]);
        let next_tau_powers_g2_1 = E::G2Prepared::from(next.tau_powers_g2[1]);

        if !same_ratio(
            power_pairs(&next.tau_powers_g1),
            (next_tau_powers_g2_0, next_tau_powers_g2_1),
        ) {
            return false;
        }

        if !same_ratio(
            power_pairs(&next.alpha_tau_powers_g1),
            (next_tau_powers_g2_0, next_tau_powers_g2_1),
        ) {
            return false;
        }

        if !same_ratio(
            power_pairs(&next.beta_tau_powers_g1),
            (next_tau_powers_g2_0, next.tau_powers_g2_1),
        ) {
            return false;
        }
        */

        *self = next;
        Ok(())
    }
}

impl<E> Default for Accumulator<E>
where
    E: PairingEngine,
{
    #[inline]
    fn default() -> Self {
        let g1_generator = E::G1Affine::prime_subgroup_generator();
        let g2_generator = E::G2Affine::prime_subgroup_generator();
        Self {
            tau_powers_g1: into_boxed_array_unchecked(
                vec![g1_generator; TAU_POWERS_G1_LENGTH].into_boxed_slice(),
            ),
            tau_powers_g2: into_boxed_array_unchecked(
                vec![g2_generator; TAU_POWERS_LENGTH].into_boxed_slice(),
            ),
            alpha_tau_powers_g1: into_boxed_array_unchecked(
                vec![g1_generator; TAU_POWERS_LENGTH].into_boxed_slice(),
            ),
            beta_tau_powers_g1: into_boxed_array_unchecked(
                vec![g1_generator; TAU_POWERS_LENGTH].into_boxed_slice(),
            ),
            beta_g2: g2_generator,
        }
    }
}

impl Serialize for Accumulator<Bls12_381> {
    #[inline]
    fn serialize<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        todo!()
    }
}

impl SerializeCompressed for Accumulator<Bls12_381> {
    #[inline]
    fn serialize_compressed<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        /*
        pub fn serialize<W: Write>(
            &self,
            writer: &mut W,
            compression: UseCompression
        ) -> io::Result<()>
        {
            fn write_all<W: Write, C: CurveAffine>(
                writer: &mut W,
                c: &[C],
                compression: UseCompression
            ) -> io::Result<()>
            {
                for c in c {
                    write_point(writer, c, compression)?;
                }

                Ok(())
            }

            write_all(writer, &self.tau_powers_g1, compression)?;
            write_all(writer, &self.tau_powers_g2, compression)?;
            write_all(writer, &self.alpha_tau_powers_g1, compression)?;
            write_all(writer, &self.beta_tau_powers_g1, compression)?;
            write_all(writer, &[self.beta_g2], compression)?;

            Ok(())
        }
            */
        todo!()
    }
}

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

///
#[inline]
fn merge_pairs<G>(lhs: &[G], rhs: &[G]) -> (G, G)
where
    G: AffineCurve,
{
    /* TODO:
    use rand::thread_rng;
    use std::sync::{Arc, Mutex};
    let chunk = (N / num_cpus::get()) + 1;
    let s = Arc::new(Mutex::new(G::Projective::zero()));
    let sx = Arc::new(Mutex::new(G::Projective::zero()));
    crossbeam::scope(|scope| {
        for (v1, v2) in lhs.chunks(chunk).zip(rhs.chunks(chunk)) {
            let s = s.clone();
            let sx = sx.clone();
            scope.spawn(move || {
                // We do not need to be overly cautious of the RNG
                // used for this check.
                let rng = &mut thread_rng();
                let mut wnaf = Wnaf::new();
                let mut local_s = G::Projective::zero();
                let mut local_sx = G::Projective::zero();
                for (v1, v2) in v1.iter().zip(v2.iter()) {
                    let rho = G::Scalar::rand(rng);
                    let mut wnaf = wnaf.scalar(rho.into_repr());
                    let v1 = wnaf.base(v1.into_projective());
                    let v2 = wnaf.base(v2.into_projective());
                    local_s.add_assign(&v1);
                    local_sx.add_assign(&v2);
                }
                s.lock().unwrap().add_assign(&local_s);
                sx.lock().unwrap().add_assign(&local_sx);
            });
        }
    });
    let s = s.lock().unwrap().into_affine();
    let sx = sx.lock().unwrap().into_affine();
    (s, sx)
    */
    todo!()
}

///
#[inline]
fn power_pairs<G, const N: usize>(points: &[G; N]) -> (G, G)
where
    G: AffineCurve,
{
    merge_pairs(&points[..(N - 1)], &points[1..])
}

///
pub const ROUNDS: usize = 89;

///
#[inline]
pub fn verify_accumulators<R>(reader: &mut R) -> Result<Accumulator<Bls12_381>, Error>
where
    R: Read,
{
    let mut accumulator = Accumulator::<Bls12_381>::default();
    let mut response_digest = ResponseDigest::default();

    for _ in 0..ROUNDS {
        let challenge_digest = accumulator.challenge_digest(response_digest);
        let next = Accumulator::deserialize(reader)?;
        let key = PublicKey::deserialize(reader)?;
        response_digest = next.response_digest(&key, challenge_digest);
        println!("{}", response_digest);
        accumulator.update(next, key, response_digest)?;
    }

    Ok(accumulator)
}

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
