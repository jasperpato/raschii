import pytest


@pytest.mark.parametrize("give_period", [True, False])
def test_infinite_depth_airy(give_period: bool):
    import raschii

    WaveModel, _AirModel = raschii.get_wave_model("Airy")

    if give_period:
        inp = dict(height=12, period=15)
    else:
        inp = dict(height=12, length=350)

    assert raschii.check_breaking_criteria(depth=5000, **inp) == ("", "")
    assert raschii.check_breaking_criteria(depth=-1, **inp) == ("", "")

    wave1 = WaveModel(depth=5000, **inp)
    wave2 = WaveModel(depth=-1, **inp)

    assert abs(wave1.length - wave2.length) < 1e-4
    assert abs(wave1.period - wave2.period) < 1e-4
    assert abs(wave1.c - wave2.c) < 1e-4


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("give_period", [True, False])
def test_infinite_depth_stokes(order: int, give_period: bool):
    import raschii

    WaveModel, _AirModel = raschii.get_wave_model("Stokes")

    if give_period:
        inp = dict(height=12, period=15)
    else:
        inp = dict(height=12, length=350)

    assert raschii.check_breaking_criteria(depth=5000, **inp) == ("", "")
    assert raschii.check_breaking_criteria(depth=-1, **inp) == ("", "")

    inp["N"] = order

    wave1 = WaveModel(depth=5000, **inp)
    wave2 = WaveModel(depth=-1, **inp)

    assert abs(wave1.length - wave2.length) < 1e-4
    assert abs(wave1.period - wave2.period) < 1e-4
    assert abs(wave1.c - wave2.c) < 1e-4


@pytest.mark.parametrize("order", [1, 5, 10])
@pytest.mark.parametrize("give_period", [True, False])
def test_infinite_depth_fenton(order: int, give_period: bool):
    import raschii

    WaveModel, _AirModel = raschii.get_wave_model("Fenton")

    if give_period:
        inp = dict(height=12, period=15)
    else:
        inp = dict(height=12, length=350)

    assert raschii.check_breaking_criteria(depth=5000, **inp) == ("", "")
    assert raschii.check_breaking_criteria(depth=-1, **inp) == ("", "")

    inp["N"] = order

    wave1 = WaveModel(depth=5000, **inp)
    wave2 = WaveModel(depth=-1, **inp)

    assert abs(wave1.length - wave2.length) < 1e-4
    assert abs(wave1.period - wave2.period) < 1e-4
    assert abs(wave1.c - wave2.c) < 1e-4


def test_fenton_velocity_acceleration_infinite_depth_matches_25_lengths():
    import numpy as np

    import raschii

    length = 100.0
    wave_inf = raschii.FentonWave(height=2.0, depth=-1, length=length, N=10)
    wave_deep = raschii.FentonWave(height=2.0, depth=25 * length, length=length, N=10)

    x = np.linspace(0, length, 7)
    # For infinite depth z=0 is the still water level, the "deep" wave has z=0 at the floor
    z_inf = np.full_like(x, -5.0)
    z_deep = z_inf + 25 * length
    for kw in (dict(all_points_wet=True), dict()):
        assert np.allclose(
            wave_inf.velocity(x, z_inf, **kw), wave_deep.velocity(x, z_deep, **kw), atol=1e-8
        )
        assert np.allclose(
            wave_inf.acceleration(x, z_inf, **kw),
            wave_deep.acceleration(x, z_deep, **kw),
            atol=1e-8,
        )

    # Above the free surface the velocity is zero without an air model
    assert np.all(wave_inf.velocity(0.0, 10.0) == 0.0)


def test_fenton_velocity_acceleration_no_overflow_deep_water():
    import numpy as np

    import raschii

    wave = raschii.FentonWave(height=1.0, depth=100.0, length=10.0, N=20)
    vel = wave.velocity(0, 99)
    acc = wave.acceleration(0, 99)
    assert np.all(np.isfinite(vel))
    assert np.all(np.isfinite(acc))


def test_fenton_num_steps_is_respected():
    import numpy as np

    from raschii.wave_fenton import wave_height_steps

    steps = wave_height_steps(4, 10.0, 100.0, 1.0)
    assert len(steps) == 4
    assert np.isclose(steps[-1], 1.0)
    assert len(wave_height_steps(1, 10.0, 100.0, 1.0)) == 1
    assert len(wave_height_steps(None, 10.0, 100.0, 1.0)) == 3


def test_fenton_stream_function_deep_water():
    import numpy as np

    import raschii

    wave = raschii.FentonWave(height=1.0, depth=100.0, length=10.0, N=20)
    assert np.all(np.isfinite(wave.stream_function(np.array([0.0, 1.0]), np.array([99.0, 99.0]))))

    length = 100.0
    wave_inf = raschii.FentonWave(height=2.0, depth=-1, length=length, N=10)
    wave_deep = raschii.FentonWave(height=2.0, depth=25 * length, length=length, N=10)
    x = np.linspace(0, length, 5)
    z = np.full_like(x, -5.0)
    assert np.allclose(wave_inf.stream_function(x, z), wave_deep.stream_function(x, z + 25 * length))
