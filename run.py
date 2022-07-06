# Do the necessary imports
from copy import deepcopy
import os
import numpy as np
from domain_model import DocumentManagement, StateVariable
from simulation import SimulationCutIn, acc_cutin_pars, acc_idm_cutin_pars, \
    IDMPlus, ACC, ACCIDMPlus, KDE, kde_from_file
from case_study import get_kpi, case_study_new, CaseStudy, sample_reactiontime, reactiontime_density

os.chdir(".")
# By default, do not overwrite previous results. Set to True to rerun all simulations.
OVERWRITE = True
model = ACC()
simulator_acc = SimulationCutIn(follower=model, follower_parameters=acc_cutin_pars,
                                min_simulation_time=5)
# simulator_acc.simulation(dict(vlead=10, vego=20, dinit=40), plot=True)
# simulator_accidm = SimulationCutIn(follower=ACCIDMPlus(),
#                                   follower_parameters=acc_idm_cutin_pars,
#                                   min_simulation_time=5)
# simulator_accidm.simulation(dict(vlead=10, vego=20, dinit=40, reactiontime=1), plot=True)
# simulator_accidm.simulation(dict(vlead=10, vego=20, dinit=40, reactiontime=1, amin=-3), plot=True)
# simulator_accidm.simulation(dict(vlead=10, vego=20, dinit=40, reactiontime=1, max_view=15), plot=True)
# Load the data with the scenarios.
scenarios = DocumentManagement(os.path.join("data", "scenarios", "cut-in.json"))
print("Number of cut-in scenarios: {:d}"
      .format(len(scenarios.collections["scenario"])))
filename = os.path.join("data", "kde", "cut-in.p")
if os.path.exists(filename) and not OVERWRITE:
    kde = kde_from_file(filename)
else:
    pars = []

    for key in scenarios.collections["scenario"]:
        scenario = scenarios.get_item("scenario", key)
        t_center = (scenario.get_tstart() + scenario.get_tend()) / 2
        vtarget, distance = scenario.get_state(scenario.get_actor_by_name("target vehicle"),
                                               StateVariable.LON_TARGET, t_center)
        vego = scenario.get_state(scenario.get_actor_by_name("ego vehicle"), StateVariable.SPEED,
                                  t_center)
        pars.append([distance, vtarget, vego])

    kde = KDE(np.array(pars))
    kde.compute_bandwidth()
    kde.pickle(filename)


# Function that checks if the scenario parameters are valid.
def check_pars(pars):
    if pars[0] <= 0 or pars[1] < 0 or pars[2] <= 0:  # dinit>0, vtarget>=0, vego>0
        return False
    if len(pars) > 3 and pars[3] <= 0:
        return False
    return True


# Function for sampling the parameters in case the ACC model is used.
def func_sample_acc():
    return kde.sample()[0]


# Function for sampling the parameters in case the ACCIDMPlus model is used.
def func_sample_accidmplus():
    return np.concatenate((kde.sample()[0], [sample_reactiontime()]))


# Function for obtaining the pdf of the parameters in case the ACC model is used.
def func_density_acc(pars):
    return kde.score_samples(pars)


# Function for obtaining the pdf of the parameters in case the ACCIDMPlus model is used.
def func_density_accidm(pars):
    return kde.score_samples(pars[:, :3]) * reactiontime_density(pars[:, 3])


default_parameters = dict(n=10000,
                          default_parameters=dict(amin=-6, k1_acc=0.9, k2_acc=0.1, k_cruise=0.2,
                                                  sensor_range=100, thw=0.21), ## input
                          percentile=2,
                          func_validity_check=check_pars,
                          func_process_result=get_kpi)
pars_acc = dict(name="cut-in_acc",
                parameters=["dinit", "vlead", "vego"],
                simulator=simulator_acc,
                func_sample=func_sample_acc,
                func_density=func_density_acc)
pars_acc.update(default_parameters)
prob_collision, sigma_collision, prob_injury, sigma_injury = case_study_new(CaseStudy(**pars_acc), overwrite=OVERWRITE)
print(prob_collision, sigma_collision, prob_injury, sigma_injury)