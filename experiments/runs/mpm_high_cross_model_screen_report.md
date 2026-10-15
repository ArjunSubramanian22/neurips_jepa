# High-excitation horizon-1 screen

Validation episodes 1440-1463. Time 0 only. Force reversal. Open loop.
Undefined cosine is null, not zero. Raw latent MSE is not compared across families.

## vicreg_action

- seed 42: cosine 0.7686335446369292 defined 24/24 one-step error 0.0977611392736435 pred norm None
- seed 43: cosine 0.7815090210694039 defined 24/24 one-step error 0.09299050271511078 pred norm None
- seed 44: cosine 0.7937686584500057 defined 24/24 one-step error 0.12042024731636047 pred norm 8.07015848217809
- seed 45: cosine 0.787760403913702 defined 24/24 one-step error 0.1153261847794056 pred norm 7.417233831067078
- seed 46: cosine 0.7836145674068771 defined 24/24 one-step error 0.10275611653923988 pred norm 7.608416936695518
- seed-level cosine medians [0.7686335446369292, 0.7815090210694039, 0.7937686584500057, 0.787760403913702, 0.7836145674068771]; median of those medians 0.783615; range 0.768634 to 0.793769.

## vicreg_noaction

- seed 42: cosine None defined 0/24 one-step error 0.5769180059432983 pred norm None
- seed 43: cosine None defined 0/24 one-step error 0.5123914331197739 pred norm 0.0
- seed 44: cosine None defined 0/24 one-step error 0.6154574453830719 pred norm 0.0
- seed 45: cosine None defined 0/24 one-step error 0.565384179353714 pred norm 0.0
- seed 46: cosine None defined 0/24 one-step error 0.5201465487480164 pred norm 0.0
- no defined cosine. Predictor contrast is the reported norm, not a cosine of 0.

## mse_action

- seed 42: cosine 0.8590980703241484 defined 24/24 one-step error 0.041707150638103485 pred norm 6.454803352713952
- seed 43: cosine 0.8724750809838043 defined 24/24 one-step error 0.046309033408761024 pred norm 6.071216352134893
- seed 44: cosine 0.8687825220521506 defined 24/24 one-step error 0.03398100100457668 pred norm 5.530642010470055
- seed 45: cosine 0.8946987970742375 defined 24/24 one-step error 0.0476661492139101 pred norm 6.856172994521115
- seed 46: cosine 0.8699151779349783 defined 24/24 one-step error 0.03440474718809128 pred norm 6.011563810532728
- seed-level cosine medians [0.8590980703241484, 0.8724750809838043, 0.8687825220521506, 0.8946987970742375, 0.8699151779349783]; median of those medians 0.869915; range 0.859098 to 0.894699.

## rssm

- seed 42: cosine 0.997282142666164 defined 24/24 one-step error 0.004911295138299465 pred norm 1.0616263068851235
- seed 43: cosine 0.9983410958341569 defined 24/24 one-step error 0.0029762386111542583 pred norm 1.1400159487664165
- seed 44: cosine 0.9993197412237649 defined 24/24 one-step error 0.0004704815219156444 pred norm 1.1427932548688704
- seed 45: cosine 0.9996931242483081 defined 24/24 one-step error 0.0009039044380187988 pred norm 1.177973072276753
- seed 46: cosine 0.9976688208973385 defined 24/24 one-step error 0.004604413639754057 pred norm 0.9830621546454026
- seed-level cosine medians [0.997282142666164, 0.9983410958341569, 0.9993197412237649, 0.9996931242483081, 0.9976688208973385]; median of those medians 0.998341; range 0.997282 to 0.999693.

## Questions

1. JEPA action seed medians: seed 42 = 0.7686335446369292, seed 43 = 0.7815090210694039, seed 44 = 0.7937686584500057, seed 45 = 0.787760403913702, seed 46 = 0.7836145674068771. Directional on every seed: True.
2. No-action: seed 42 defined 0/24 pred norm None, seed 43 defined 0/24 pred norm 0.0, seed 44 defined 0/24 pred norm 0.0, seed 45 defined 0/24 pred norm 0.0, seed 46 defined 0/24 pred norm 0.0. Cosine stays null with defined count 0: True.
3. MSE action seed medians, compared only inside the MSE latent: seed 42 = 0.8590980703241484, seed 43 = 0.8724750809838043, seed 44 = 0.8687825220521506, seed 45 = 0.8946987970742375, seed 46 = 0.8699151779349783. Defined on every seed: True.
4. RSSM seed medians in concat(h, stochastic mean), from one shared filtered posterior: seed 42 = 0.997282142666164, seed 43 = 0.9983410958341569, seed 44 = 0.9993197412237649, seed 45 = 0.9996931242483081, seed 46 = 0.9976688208973385. Defined on every seed: True.
5. The directional effect is present in JEPA, MSE, and RSSM, and absent for no-action: True.
6. Seed variation is the range of the five seed medians printed above. The 24 episodes inside one seed are not five training replicates.
