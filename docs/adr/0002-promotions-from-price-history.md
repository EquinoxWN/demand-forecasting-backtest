# ADR 0002: Infer promotions from each item's price history

- **Status:** Accepted

## Context

The backlog asks for promotion features, but M5 has no promotion flag. It has weekly shelf prices
per item and store (`sell_prices.csv`), SNAP days per state and named events. A discount has to be
measured against some "regular" price, and that reference must not use future prices, or the
feature leaks information a forecaster would not have.

## Decision

- The regular price of an item in a store is the highest weekly price seen so far (a running
  maximum over weeks, `np.fmax.accumulate`), never a full-history maximum or median.
- `discount = 1 - price / regular_price`; a week with `discount >= 0.05` is a promotion (`promo`).
- Weeks without a price row mean the item is not on the shelf: `on_shelf = 0`, discount 0.
- At store and total level, promotions are summarised as the share of on-shelf items on promotion
  and their mean discount.

## Consequences

- Features never look ahead; a test changes later prices and sales and checks that no earlier
  value moves.
- A permanent price cut is treated as a lasting promotion until the price rises again, and a
  price rise sets a new, higher reference. Promotions that do not change the shelf price (displays,
  coupons) are invisible.
- The 5% threshold is a parameter (`PROMO_THRESHOLD`); its effect on model accuracy can only be
  measured once models exist (M2).
- Alternative rejected: the median price over the whole history as reference. It is more robust
  to one-off price spikes but uses future weeks.
