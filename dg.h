#ifndef __dg_h
#define __dg_h

#include <base/quadrature_lib.h>
#include <base/function.h>
#include <lac/vector.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/grid_out.h>
#include <grid/grid_refinement.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <fe/fe_values.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <numerics/data_out.h>
#include <numerics/vector_tools.h>
#include <fe/mapping_q1.h>
		
#include <fe/fe_dgq.h>
				 
#include <numerics/derivative_approximation.h>
#include <numerics/solution_transfer.h>

#include <meshworker/dof_info.h>
#include <meshworker/integration_info.h>
#include <meshworker/simple.h>
#include <meshworker/loop.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <boost/python/dict.hpp>
#include <boost/python.hpp>
namespace bp = boost::python;
//------------------------------------------------------------------------------
// Main class of the problem
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Class for integrating rhs using MeshWorker
//------------------------------------------------------------------------------
template <int dim>
class RHSIntegrator
{
   public:
      RHSIntegrator (const dealii::DoFHandler<dim>& dof_handler)
         : dof_info (dof_handler) {};

      dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
      dealii::MeshWorker::DoFInfo<dim> dof_info;
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> >
         assembler;
};

template<int dim>
class ExactSolution: public dealii::Function<dim>
{
    public:
        ExactSolution(const double time) { t = time; }   
        virtual double value(const dealii::Point<dim> &p,
                             const unsigned int component) const;
        double t;
};

template <int dim>
class Step12
{
   public:
      Step12 (bp::dict params);
      void run ();
      
   private:
      void setup_system();
      void assemble_mass_matrix();
      void set_initial_condition();
      void setup_mesh_worker(RHSIntegrator<dim>&);
      void assemble_rhs(RHSIntegrator<dim>&);
      void compute_dt();
      void solve();
      void step(RHSIntegrator<dim>& rhs_integrator, int level,
                double local_dt, dealii::Vector<double>& coarse_source);
      void refine_grid(dealii::Vector<double> refinement_soln);
      void output_results(const unsigned int cycle,
                          const double time) const;
      void compute_error(const double time) const;
      
      dealii::Triangulation<dim> triangulation;
      const dealii::MappingQ1<dim> mapping;
   
      unsigned int degree;
      dealii::FE_DGQArbitraryNodes<dim> fe;
      dealii::DoFHandler<dim> dof_handler;
      
      dealii::Vector<double> inv_mass_matrix;
      
      dealii::Vector<double> predictor;
      dealii::Vector<double> solution;
      dealii::Vector<double> solution_old;
      dealii::Vector<double> right_hand_side;
      int current_level;
      double dt;
      double cfl;
      double C;

      int max_level;
      int min_level;
      double epsilon;
      double chi_refine;
      double chi_coarse;
   
      typedef dealii::MeshWorker::DoFInfo<dim> DoFInfo;
      typedef dealii::MeshWorker::IntegrationInfo<dim> CellInfo;
      
      static void integrate_cell_term (DoFInfo& dinfo, CellInfo& info);
      static void integrate_boundary_term (DoFInfo& dinfo, CellInfo& info);
      static void integrate_face_term (DoFInfo& dinfo1, DoFInfo& dinfo2,
                                       CellInfo& info1, CellInfo& info2);
};
#endif
