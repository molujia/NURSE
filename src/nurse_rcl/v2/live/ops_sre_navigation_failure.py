"""New-profile failure metadata; legacy failure serialization is unchanged."""

from __future__ import annotations

from typing import Literal, cast, get_args

from pydantic import Field

from nurse_rcl.v2.exchange_json import ExchangeBoundaryError
from nurse_rcl.v2.live.ops_failure import FailureStage, OpsFailureDetail, failure_detail

NavigationErrorCode = Literal[
    "sre_question_not_previously_visible",
    "sre_response_not_previously_visible",
    "sre_latest_result_not_addressed",
]


class OpsNavigationFailureDetail(OpsFailureDetail):
    decision_index: int | None = Field(default=None, ge=1, le=8, strict=True)
    exchange_code: NavigationErrorCode | None = None


def navigation_failure(
    stage: FailureStage,
    exc: BaseException,
    *,
    decision_index: int | None,
    detail: OpsFailureDetail | None = None,
) -> OpsNavigationFailureDetail:
    base = detail if detail is not None else failure_detail(stage, exc)
    code = exc.code if isinstance(exc, ExchangeBoundaryError) else None
    return OpsNavigationFailureDetail(
        **base.model_dump(),
        decision_index=decision_index,
        exchange_code=cast(NavigationErrorCode, code)
        if code in get_args(NavigationErrorCode)
        else None,
    )
