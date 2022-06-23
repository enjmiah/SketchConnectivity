#include "Context.h"

Context::Context(bool grb_env, bool grb_log) {
	if (grb_env) {
		grb_ptr = std::make_unique<GRBEnv>(true);
		if (!grb_log) {
			grb_ptr->set(GRB_IntParam_LogToConsole, 0);
		}
		try {
			grb_ptr->start();
		}
		catch (GRBException e) {
			std::cout << "Error code = " << e.getErrorCode() << std::endl;
			std::cout << e.getMessage() << std::endl;
			throw e;
		}
	}
}

void Context::optimize_model(GRBModel* model) const {
	model->set(GRB_DoubleParam_TimeLimit, 120);

	try {
		//model.set(GRB_DoubleParam_FeasibilityTol, 1e-2);
		model->optimize();
	}
	catch (GRBException e) {
		try {
			//model.set(GRB_IntParam_DualReductions, 0);
			model->set(GRB_IntParam_BarHomogeneous, 1);
			model->set(GRB_DoubleParam_PSDTol, 1e-4);
			//model->set(GRB_IntParam_NonConvex, 2);
			model->optimize();
		}
		catch (GRBException e2) {
			if (e2.getErrorCode() == 10020) {
				try {
					//model.set(GRB_IntParam_DualReductions, 0);
					model->set(GRB_IntParam_BarHomogeneous, 1);
					model->set(GRB_DoubleParam_PSDTol, 1e-4);
					model->set(GRB_IntParam_NonConvex, 2);
					model->optimize();
				}
				catch (GRBException e3) {
					std::cout << "Error code = " << e3.getErrorCode() << std::endl;
					std::cout << e3.getMessage() << std::endl;
					throw e3;
				}
			}
			else {
				std::cout << "Error code = " << e2.getErrorCode() << std::endl;
				std::cout << e2.getMessage() << std::endl;
				throw e2;
			}
		}
	}
	if (model->get(GRB_IntAttr_Status) == GRB_NUMERIC) {
		//model.set(GRB_IntParam_DualReductions, 0);
		model->set(GRB_IntParam_BarHomogeneous, 1);
		model->optimize();
	}
	if (model->get(GRB_IntAttr_Status) == GRB_NUMERIC) {
		//model->set(GRB_IntParam_BarHomogeneous, -1);
		model->set(GRB_IntParam_ScaleFlag, 2);
		model->set(GRB_DoubleParam_ObjScale, -0.5);
		model->optimize();
	}
}
