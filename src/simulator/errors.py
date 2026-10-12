"""Exception hierarchy for the simulator.

Every error a user can trigger through normal use derives from
:class:`SimulatorError`, so the UI can catch one base class and show a
friendly message while genuine bugs still surface as ordinary exceptions.
"""

from __future__ import annotations


class SimulatorError(Exception):
    """Base class for all expected, user-facing simulator errors."""


class OrderValidationError(SimulatorError):
    """An order was rejected before execution."""


class InvalidQuantityError(OrderValidationError):
    """Quantity is not a positive whole number of shares."""


class InsufficientCashError(OrderValidationError):
    """Not enough cash to pay for a buy, including fees."""


class InsufficientSharesError(OrderValidationError):
    """Selling more shares than are held (short selling is not allowed)."""


class InvalidOrderError(OrderValidationError):
    """Order parameters are inconsistent (e.g. a limit order without a limit price)."""


class NoPriceDataError(SimulatorError):
    """No price data exists for the requested ticker and date."""


class UnknownTickerError(NoPriceDataError):
    """The ticker is not present in the price store at all."""
