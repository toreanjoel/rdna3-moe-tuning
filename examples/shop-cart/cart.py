"""A small shopping cart. Has bugs and a missing feature on purpose: see README.md."""
from dataclasses import dataclass, field


@dataclass
class Item:
    sku: str
    price_cents: int
    qty: int = 1


@dataclass
class Cart:
    items: dict = field(default_factory=dict)
    coupon: str | None = None

    def add(self, sku: str, price_cents: int, qty: int = 1) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if sku in self.items:
            self.items[sku].qty = qty          # bug: should add to the existing quantity
        else:
            self.items[sku] = Item(sku, price_cents, qty)

    def remove(self, sku: str, qty: int = 1) -> None:
        item = self.items[sku]
        item.qty -= qty
        if item.qty < 0:                       # bug: an item at 0 should be removed too
            del self.items[sku]

    def subtotal(self) -> int:
        return sum(i.price_cents for i in self.items.values())   # bug: ignores qty

    def apply_coupon(self, code: str) -> None:
        # TODO: "SAVE10" = 10% off the subtotal, "FLAT500" = 500 cents off (never below 0); anything else raises ValueError
        raise NotImplementedError

    def total(self) -> int:
        # TODO: subtotal minus the coupon discount, rounded to the nearest cent
        return self.subtotal()
