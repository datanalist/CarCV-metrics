import numpy as np
import prep_stanford_cars


def test_build_stanford_records_joins_class_names():
    class_names = ["Acura RL Sedan 2012", "Audi TT Coupe 2012"]
    annos = [("00001.jpg", 1), ("00002.jpg", 2), ("00003.jpg", 1)]
    recs = prep_stanford_cars.build_stanford_records(class_names, annos)
    assert recs == [
        {"file_name": "00001.jpg", "label": "Acura RL Sedan 2012"},
        {"file_name": "00002.jpg", "label": "Audi TT Coupe 2012"},
        {"file_name": "00003.jpg", "label": "Acura RL Sedan 2012"},
    ]


def test_parse_mat_class_names_and_annos(tmp_path):
    from scipy.io import savemat
    savemat(tmp_path / "meta.mat",
            {"class_names": np.array([["Acura RL Sedan 2012", "Audi TT Coupe 2012"]],
                                     dtype=object)})
    annos = np.zeros((2,), dtype=[("class", "O"), ("fname", "O")])
    annos[0] = (np.array([[1]]), np.array(["00001.jpg"]))
    annos[1] = (np.array([[2]]), np.array(["00002.jpg"]))
    savemat(tmp_path / "annos.mat", {"annotations": annos})

    names = prep_stanford_cars.parse_class_names(tmp_path / "meta.mat")
    assert names[0] == "Acura RL Sedan 2012"
    pairs = prep_stanford_cars.parse_annos(tmp_path / "annos.mat")
    assert pairs[0] == ("00001.jpg", 1)
