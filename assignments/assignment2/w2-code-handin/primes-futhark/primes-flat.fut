-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 } auto output

let mkFlagArray 't [m] (aoa_shp: [m]i64) 
                       (zero: t) 
                       (aoa_val: [m]t) 
                       (aoa_len: i64) : [aoa_len]t =
  let shp_rot = map (\i -> if i == 0 then 0
                           else aoa_shp[i - 1]) (iota m)
  let shp_scn = scan (+) 0 shp_rot
  let shp_ind = map2 (\shp ind -> if shp == 0 then -1
                                else ind) aoa_shp shp_scn
  in scatter (replicate aoa_len zero) shp_ind aoa_val

let sgmSumInc [n] (flg : [n]bool) (arr : [n]i64) : [n]i64 =
  let flgs_vals = 
    scan (\ (i1, x1) (i2,x2) -> 
            let i = i1 || i2 in
              if i2 then (i, x2)
              else (i, x1 + x2))
         (false, 0i64) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      let composite = 
        let iot = 
          let flag = mkFlagArray mult_lens 0i64 mult_lens flat_size
          let flag2 = map (\f -> f != 0) flag
          let vals = map (\f -> if f != 0 then 0 else 1) flag
          in sgmSumInc flag2 vals

        let arr = map (+2) iot

        let ps = 
          let (flag_n, flag_v) =
                            unzip <|
                            mkFlagArray mult_lens (0,0) (zip mult_lens sq_primes) flat_size
          let flag_n2 = map (\f -> f != 0) flag_n
          in sgmSumInc flag_n2 flag_v
        
        in map2 (*) ps arr
      --------------------------------------------------------------
      -- The current iteration knowns the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the right `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
      let not_primes = composite

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

      let zero_array = replicate flat_size 0i8
      let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
      let prime_flags= scatter mostly_ones not_primes zero_array
      let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                            (0...len)

      in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
-- $ futhark opencl primes-flat.fut
-- $ echo "10000000" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
let main (n : i64) : []i64 = primesFlat n
