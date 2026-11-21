import unittest

from cart import Cart


class CartTest(unittest.TestCase):
    def test_add_accumulates(self):
        c = Cart(); c.add("a", 100, 2); c.add("a", 100, 3)
        self.assertEqual(c.items["a"].qty, 5)

    def test_subtotal_uses_qty(self):
        c = Cart(); c.add("a", 250, 2); c.add("b", 100)
        self.assertEqual(c.subtotal(), 600)

    def test_remove_to_zero_drops_item(self):
        c = Cart(); c.add("a", 100, 2); c.remove("a", 2)
        self.assertNotIn("a", c.items)

    def test_bad_qty(self):
        with self.assertRaises(ValueError):
            Cart().add("a", 100, 0)

    def test_save10(self):
        c = Cart(); c.add("a", 999); c.apply_coupon("SAVE10")
        self.assertEqual(c.total(), 899)

    def test_flat500_floor(self):
        c = Cart(); c.add("a", 300); c.apply_coupon("FLAT500")
        self.assertEqual(c.total(), 0)

    def test_unknown_coupon(self):
        with self.assertRaises(ValueError):
            Cart().apply_coupon("NOPE")


if __name__ == "__main__":
    unittest.main()
