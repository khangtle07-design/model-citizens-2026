# Write-up

Our model treats the Kalshi market as a strong but imperfect forecaster and looks for the few races
where an independent forecast disagrees with it enough to be worth a pick. The independent forecast
is a fundamentals regression trained on every House race from 2004 to 2024 (lagged margin,
incumbency, national swing; leave-one-cycle-out accuracy 89–97% per cycle), applied to 2026 using a
D+8.4 generic ballot. For the Senate it is precision-weighted against nonpartisan polling averages,
and for the House it is blended with Cook, Sabato and Inside Elections ratings. We then
average it with the market in log-odds (equal weight in the Senate, 35% model in the House) and
simulate 20,000 elections with a shared national error, so that polling misses hit many races at
once. The scoring rule makes picking underdogs mathematically attractive even with no information:
if Kalshi is calibrated, picking every underdog doubles expected points while getting about 50 of
470 races right. We reject that as a guess, not a forecast, and allow a pick only on a side the
model gives at least a 40% chance. Within that constraint we choose the rule that maximises
mean − 0.5·SD of simulated points. Data: MIT MEDSL results, 2024 presidential results by
district, Silver Bulletin, VoteHub and RealClearPolling averages, the three rating shops and Kalshi
prices. Partisan internals and promotional market blogs are excluded. Risks: the blend weights are
judgment calls with no historical market data to fit them; redistricted seats rely on estimated
district lean; the six against-market picks are correlated (four of them back Republicans against
the market), so they tend to hit or miss together; and the market prices will move
before the Nov 1 benchmark, which can change picks.
