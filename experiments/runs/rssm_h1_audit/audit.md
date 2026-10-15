# RSSM horizon-1 audit

Validation episodes 1440-1463. Time 0. Force reversal unless a control says otherwise.
Test episodes 1620-1799 were not read.

One-step observe and imagine apply the same GRU update before the posterior sees the frame. concat(h, z) therefore shares delta_h exactly. The corrected predictor contrast is the stochastic mean only.

Primary vector after this audit: `stochastic_mean_z`.

## seed 42

- deterministic_h: cosine 1.0 pred 1.04551 target 1.04551 ratio 0.9999999999990434
- stochastic_mean_z: cosine 0.9143820898398167 pred 0.163345 target 0.186616 ratio 0.8568987385347472
- concat_h_z: cosine 0.997282142666164 pred 1.06163 target 1.06919 ratio 0.9961265942250574
- same-action z pred norm 0; target norm 0
- shuffle vs original target, z cosine 0.3657340215077771
- random action with its own physical target, z cosine 0.4097778520765789
- bitwise shared h 1.0; branch copies equal 1.0; recorded prior/posterior z MSE 0.0245565

## seed 43

- deterministic_h: cosine 1.0 pred 1.1158 target 1.1158 ratio 0.9999999999991033
- stochastic_mean_z: cosine 0.9653611693987317 pred 0.233738 target 0.267151 ratio 0.9089286567984431
- concat_h_z: cosine 0.9983410958341569 pred 1.14002 target 1.14788 ratio 0.9960326912131898
- same-action z pred norm 0; target norm 0
- shuffle vs original target, z cosine 0.3436207028709247
- random action with its own physical target, z cosine 0.7247325190340561
- bitwise shared h 1.0; branch copies equal 1.0; recorded prior/posterior z MSE 0.0148812

## seed 44

- deterministic_h: cosine 1.0 pred 1.12349 target 1.12349 ratio 0.9999999999991094
- stochastic_mean_z: cosine 0.9817829424739709 pred 0.202837 target 0.243662 ratio 0.9690889093638573
- concat_h_z: cosine 0.9993197412237649 pred 1.14279 target 1.1511 ratio 0.9989823912392354
- same-action z pred norm 0; target norm 0
- shuffle vs original target, z cosine 0.44608863724423065
- random action with its own physical target, z cosine 0.9370770953033043
- bitwise shared h 1.0; branch copies equal 1.0; recorded prior/posterior z MSE 0.00235241

## seed 45

- deterministic_h: cosine 1.0 pred 1.16414 target 1.16414 ratio 0.99999999999914
- stochastic_mean_z: cosine 0.9925980127589035 pred 0.202548 target 0.208747 ratio 0.984201045545968
- concat_h_z: cosine 0.9996931242483081 pred 1.17797 target 1.17803 ratio 0.9994651433989118
- same-action z pred norm 0; target norm 0
- shuffle vs original target, z cosine 0.458499528570281
- random action with its own physical target, z cosine 0.8043241298696351
- bitwise shared h 1.0; branch copies equal 1.0; recorded prior/posterior z MSE 0.00451952

## seed 46

- deterministic_h: cosine 1.0 pred 0.96301 target 0.96301 ratio 0.9999999999989594
- stochastic_mean_z: cosine 0.9409360534618483 pred 0.182878 target 0.20989 ratio 0.9041948694924938
- concat_h_z: cosine 0.9976688208973385 pred 0.983062 target 0.994242 ratio 0.996798480646591
- same-action z pred norm 0; target norm 0
- shuffle vs original target, z cosine 0.45256013945813084
- random action with its own physical target, z cosine 0.7538952721095351
- bitwise shared h 1.0; branch copies equal 1.0; recorded prior/posterior z MSE 0.0230221

## Physical fork

- median action L2 1.00738 (min 0.350218)
- median position RMS 0.00134647 (min 0.000370225)
- median raster MSE 0.00112335
- median identical-raster fraction 0.855103
