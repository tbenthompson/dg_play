/* $Id: step-12.cc 22886 2010-11-29 23:48:38Z bangerth $ */
/* Author: Guido Kanschat, Texas A&M University, 2009 */

/*    $Id: step-12.cc 22886 2010-11-29 23:48:38Z bangerth $       */
/*                                                                */
/*    Copyright (C) 2010 by the deal.II authors */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

// Modifications by Praveen. C, http://math.tifrbng.res.in/~praveen
// Explicit time-stepping Runge-Kutta DG method
// Mass matrix on each cell is computed, inverted and the inverse 
// is stored. Then in each time iteration, we need to compute right
// hand side and multipy by inverse mass mass matrix. After that
// solution is advanced to new time level by an RK scheme.

#include "dg.h"

using namespace dealii;

//RK43 from the Schlegel 2009 paper
double rk_c[5] = {0.0, 0.5, 0.5, 1.0, 1.0};
double rk_ab[5][4] = {{0.0, 0.0, 0.0, 0.0},
                     {0.5, 0.0, 0.0, 0.0},
                     {-1.0 / 6.0, 2.0 / 3.0, 0.0, 0.0},
                     {1.0 / 3.0, -1.0 / 3.0, 1.0, 0.0},
                     {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0}};


template<int dim>
double ExactSolution<dim>::value(const dealii::Point<dim> &p,
                             const unsigned int component = 0) const
{
    double angle = t;
    double center_x = 0.5 * std::cos(angle);
    double center_y = 0.5 * std::sin(angle);
    double r2 = std::pow(p(0) - center_x, 2.0) + std::pow(p(1) - center_y, 2.0);
    return std::exp(-r2 * 100);
}

//------------------------------------------------------------------------------
// Speed of advection
//------------------------------------------------------------------------------
template <int dim>
void advection_speed(const Point<dim>& p, Point<dim>& v)
{
   // v(0) = -1.0;-p(1);
   // v(1) = 0.0;//p(0);
   v(0) = -p(1);
   v(1) = p(0);
}
//------------------------------------------------------------------------------
// Initial condition function class
//------------------------------------------------------------------------------
template <int dim>
class InitialCondition: public Function<dim>
{
public:
   InitialCondition () {};
   virtual void value_list (const std::vector<Point<dim> > &points,
                            std::vector<double> &values,
                            const unsigned int component=0) const;
};

// Computes boundary condition value at a list of boundary points
template <int dim>
void InitialCondition<dim>::value_list(const std::vector<Point<dim> > &points,
                                     std::vector<double> &values,
                                     const unsigned int) const
{
   Assert(values.size()==points.size(),
          ExcDimensionMismatch(values.size(),points.size()));
   
   for (unsigned int i=0; i<values.size(); ++i)
   {
      double r2 = std::pow(points[i](0)-0.5, 2.0) + std::pow(points[i](1), 2.0);
      values[i] = std::exp(-r2 * 100);
   }
}

//------------------------------------------------------------------------------
// Boundary condition function class
//------------------------------------------------------------------------------
template <int dim>
class BoundaryValues: public Function<dim>
{
  public:
    BoundaryValues () {};
    virtual void value_list (const std::vector<Point<dim> > &points,
			                    std::vector<double> &values,
			                    const unsigned int component=0) const;
};

// Computes boundary condition value at a list of boundary points
template <int dim>
void BoundaryValues<dim>::value_list(const std::vector<Point<dim> > &points,
				       std::vector<double> &values,
				       const unsigned int) const
{
   Assert(values.size()==points.size(),
          ExcDimensionMismatch(values.size(),points.size()));
   
   for (unsigned int i=0; i<values.size(); ++i)
   {
      values[i]=0.0;
   }
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <int dim>
Step12<dim>::Step12 (bp::dict params)
      :
      mapping (),
      degree (bp::extract<int>(params["degree"])),
      fe (QGaussLobatto<1>(degree + 1)),
      dof_handler (triangulation)
{ 
   cfl = 1.0 / (2.0 * degree + 1.0);
   max_level = bp::extract<int>(params["max_level"]);
   min_level = bp::extract<int>(params["min_level"]);
   epsilon = bp::extract<double>(params["epsilon_refine"]);
   chi_refine = bp::extract<double>(params["chi_refine"]);
   chi_coarse = bp::extract<double>(params["chi_coarse"]);
}

//------------------------------------------------------------------------------
// Make dofs and allocate memory
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::setup_system ()
{
    /* std::cout << "Allocating memory ...\n"; */

    dof_handler.distribute_dofs (fe);
   
    inv_mass_matrix.reinit(dof_handler.n_dofs());
    predictor.reinit(dof_handler.n_dofs());
    solution.reinit (dof_handler.n_dofs());
    solution_old.reinit (dof_handler.n_dofs());
    right_hand_side.reinit (dof_handler.n_dofs());
}

//------------------------------------------------------------------------------
// Assemble mass matrix for each cell
// Invert it and store
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::assemble_mass_matrix ()
{
   // std::cout << "Constructing mass matrix ...\n";
   
   QGaussLobatto<dim>  quadrature_formula(fe.degree+1);
   
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values | update_JxW_values);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
   
   // Cell iterator
   typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (unsigned int c = 0; cell!=endc; ++cell, ++c)
    {
        fe_values.reinit (cell);
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            inv_mass_matrix[local_dof_indices[q_point]] =
                               1.0 / fe_values.JxW(q_point);
    }
   
}
//------------------------------------------------------------------------------
// Project initial condition
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::set_initial_condition ()
{
    VectorTools::create_right_hand_side(dof_handler,
                                       QGaussLobatto<dim>(fe.degree + 1),
                                       InitialCondition<dim>(),
                                       solution);
    solution.scale(inv_mass_matrix);
}
//------------------------------------------------------------------------------
// Create mesh worker for integration
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::setup_mesh_worker (RHSIntegrator<dim>& rhs_integrator)
{   
   std::cout << "Setting up mesh worker ...\n";

   MeshWorker::IntegrationInfoBox<dim>& info_box = rhs_integrator.info_box;
   MeshWorker::DoFInfo<dim>& dof_info = rhs_integrator.dof_info;
   MeshWorker::Assembler::ResidualSimple< Vector<double> >&
      assembler = rhs_integrator.assembler;

   const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
   info_box.cell_quadrature = QGaussLobatto<dim>(n_gauss_points);
   info_box.face_quadrature = QGaussLobatto<dim-1>(n_gauss_points);
   info_box.boundary_quadrature = QGaussLobatto<dim-1>(n_gauss_points);

   // Add solution vector to info_box
   NamedData< Vector<double>* > solution_data;
   solution_data.add (&solution, "solution");
   info_box.cell_selector.add     ("solution", true, false, false);
   info_box.boundary_selector.add ("solution", true, false, false);
   info_box.face_selector.add     ("solution", true, false, false);
   
   info_box.initialize_update_flags ();
   info_box.add_update_flags_all      (update_quadrature_points);
   info_box.add_update_flags_cell     (update_gradients);
   info_box.add_update_flags_boundary (update_values);
   info_box.add_update_flags_face     (update_values);

   info_box.initialize (fe, mapping, solution_data);
   
   // Attach rhs vector to assembler
   NamedData< Vector<double>* > rhs;
   Vector<double>* data = &right_hand_side;
   rhs.add (data, "RHS");
   assembler.initialize (rhs);
}

//------------------------------------------------------------------------------
// Compute time-step
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::compute_dt()
{
    // std::cout << "Computing local time-step ...\n";
    dt = 1.0e20;
    dt_max = 0.0;
    
    // The tree root is set
    TimeStepLevelTree dt_root;
    dt_root.dt = 1.0;
    dt_root.index = 1;

    // Cell iterator
    typename DoFHandler<dim>::active_cell_iterator 
        cell = dof_handler.begin_active(),
             endc = dof_handler.end();
    for (unsigned int c = 0; cell!=endc; ++cell, ++c)
    {
        const double h = cell->diameter ();

        const Point<dim> cell_center = cell->center();
        Point<dim> beta;
        advection_speed(cell_center, beta);

        const double local_dt = h / beta.norm();

        for(int op = 0; op < dt_options.size();op++)
        {
            if (dt_options[op] < local_dt)
            {
                dt_indices[op] = cell->index();
            }
        }

        dt = std::min(dt, local_dt);
        dt_max = std::max(dt_max, local_dt[c]);
    }
    dt *= cfl;
}

//------------------------------------------------------------------------------
// Compute flux -- not a member function
//------------------------------------------------------------------------------
template <int dim>
Point<dim> flux_function(Point<dim> pt, double soln_val)
{
    Point<dim> speed;
    advection_speed(pt, speed);
    return speed * soln_val;
}


//------------------------------------------------------------------------------
// Assemble rhs of the problem
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::assemble_rhs (RHSIntegrator<dim>& rhs_integrator)
{
   right_hand_side = 0.0;

   MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>,
                    MeshWorker::IntegrationInfoBox<dim> >
      (dof_handler.begin_active(current_level), 
       dof_handler.end_active(current_level),
       rhs_integrator.dof_info, 
       rhs_integrator.info_box,
       &Step12<dim>::integrate_cell_term,
       &Step12<dim>::integrate_boundary_term,
       &Step12<dim>::integrate_face_term,
       rhs_integrator.assembler, true);

   // Multiply by inverse mass matrix
    right_hand_side.scale(inv_mass_matrix);
}

//------------------------------------------------------------------------------
// Compute cell integral
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::integrate_cell_term (DoFInfo& dinfo, CellInfo& info)
{
    const FEValuesBase<dim>& fe_v  = info.fe_values();
    const std::vector<double>& sol = info.values[0][0];
    Vector<double>& local_vector   = dinfo.vector(0).block(0);
    const std::vector<double>& JxW = fe_v.get_JxW_values ();

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {
            local_vector(i) += 
                flux_function(fe_v.quadrature_point(point), sol[point]) * 
                fe_v.shape_grad(i, point) * JxW[point];
        }
    }
}

//------------------------------------------------------------------------------
// Compute boundary integral
//------------------------------------------------------------------------------
    template <int dim>
void Step12<dim>::integrate_boundary_term (DoFInfo& dinfo, CellInfo& info)
{
    const FEValuesBase<dim>& fe_v  = info.fe_values();
    const std::vector<double>& sol = info.values[0][0];

    Vector<double>& local_vector = dinfo.vector(0).block(0);

    const std::vector<double>& JxW = fe_v.get_JxW_values ();
    const std::vector<Point<dim> >& normals = fe_v.get_normal_vectors ();

    std::vector<double> g(fe_v.n_quadrature_points);

    static BoundaryValues<dim> boundary_function;
    boundary_function.value_list (fe_v.get_quadrature_points(), g);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        Point<dim> beta;
        advection_speed(fe_v.quadrature_point(point), beta);
        const double beta_n = beta * normals[point];
        if (beta_n > 0)
        {
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            {
                local_vector(i) -= 
                    flux_function(fe_v.quadrature_point(point), sol[point]) * 
                    normals[point] *
                    fe_v.shape_value(i, point) *
                    JxW[point];
            }
        }
        else
        {
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            {
                local_vector(i) -=
                    flux_function(fe_v.quadrature_point(point), g[point]) * 
                    normals[point] *
                    fe_v.shape_value(i, point) *
                    JxW[point];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Compute integral over internal faces
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::integrate_face_term(DoFInfo& dinfo1, DoFInfo& dinfo2,
				      CellInfo& info1, CellInfo& info2)
{
    const FEValuesBase<dim>& fe_v          = info1.fe_values();
    const FEValuesBase<dim>& fe_v_neighbor = info2.fe_values();

    //Check if we should build this term right now...
    //If the other cell is from a coarser level, then we build the term.
    //If the other cell is from the same level, then we build the term.
    //If the other cell is from a finer level, do not build the term.
    unsigned int here_level = fe_v.get_cell()->level();
    unsigned int other_level = fe_v_neighbor.get_cell()->level();
    if (here_level < other_level)
    {
        std::cout << "Skipping face" << std::endl;
        return;
    }

    const std::vector<double>& sol1 = info1.values[0][0];
    const std::vector<double>& sol2 = info2.values[0][0];

    Vector<double>& local_vector1 = dinfo1.vector(0).block(0);
    Vector<double>& local_vector2 = dinfo2.vector(0).block(0);

    const std::vector<double>& JxW = fe_v.get_JxW_values();
    const std::vector<Point<dim> >& normals = fe_v.get_normal_vectors ();
    for (unsigned int point=0; point < fe_v.n_quadrature_points; ++point)
    {
        Point<dim> beta;
        advection_speed(fe_v.quadrature_point(point), beta);

        // An upwind flux?
        const double C = std::abs(normals[point][0] * beta[0] +
                                  normals[point][1] * beta[1]);
        Point<dim> f_star_1 = flux_function(fe_v.quadrature_point(point), sol1[point]);
        Point<dim> f_star_2 = flux_function(fe_v.quadrature_point(point), sol2[point]);
        Point<dim> avg_flux = f_star_1 + f_star_2;
        Point<dim> jump_flux = C * normals[point] * (sol1[point] - sol2[point]);
        double flux = 0.5 * (avg_flux + jump_flux) * normals[point] * JxW[point];
        for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        {
            local_vector1(i) -= flux *
                fe_v.shape_value(i, point);
        }

        for (unsigned int k = 0; k < fe_v_neighbor.dofs_per_cell; ++k)
        {
            local_vector2(k) += flux *
                fe_v_neighbor.shape_value(k, point);
        }
    }
}

//------------------------------------------------------------------------------
// Solve the problem to convergence by RK time integration
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::solve ()
{
    RHSIntegrator<dim> rhs_integrator (dof_handler);
    setup_mesh_worker (rhs_integrator);

    std::cout << "Solving by RK ...\n";

    unsigned int iter = 0;
    double time = 0;
    compute_dt();
    Vector<double> zero;
    zero.reinit(dof_handler.n_dofs());
    zero = 0;

    while (time < 10 * M_PI && iter < 10000)
    {
        solution_old = solution;

        step(rhs_integrator, min_level, dt, zero);

        predictor = solution;
        predictor.sadd(2.0, -1.0, solution_old);

        ++iter; time += dt;
        if(std::fmod(iter, 10) == 0) 
        {
            std::cout << "Iterations=" << iter 
                << ", t = " << time << endl;
            output_results(iter, time);
            compute_error(time);
        }
        if(std::fmod(iter, 3) == 0)
        {
            refine_grid(predictor);
            compute_dt();
            Vector<double> zero;
            zero.reinit(dof_handler.n_dofs());
            zero = 0;
        }
    }
}

template <int dim>
void Step12<dim>::step(RHSIntegrator<dim>& rhs_integrator,
                       int level,
                       double local_dt,
                       Vector<double>& coarse_source)
{
    std::vector<Vector<double> > k(4);
    Vector<double> r;
    r.reinit(dof_handler.n_dofs());

    for(int stage = 1; stage < 5; stage++)
    {
        current_level = level;
        assemble_rhs(rhs_integrator);
        k[stage - 1] = right_hand_side;

        r = 0;
        for (int prior_stage = 0; prior_stage < stage; prior_stage++)
        {
            // The reformulation of the Runge Kutta method in Schlegel 2009
            // is in terms of the "squiggle" matrix
            const double a_squiggle = rk_ab[stage][prior_stage] -
                                      rk_ab[stage - 1][prior_stage];
            r.add(a_squiggle * local_dt, k[prior_stage]);
        }

        // See page 3 of the Schlegel paper for this definition.
        const double c_squiggle = rk_c[stage] - rk_c[stage - 1];

        // If this is a coarse-only step, then just add the runge kutta 
        // step and move to the next stage. 
        // If this is a coarse and fine step, then recursive to the next 
        // time step level.
        if (c_squiggle > 0 && level < max_level)
        {
            r.scale(1.0 / c_squiggle);
            step(rhs_integrator, level + 1, local_dt / 2.0, r);
        } else
        {
            solution.add(r);
        }
    }
}

//---------------------------------------------------------------------
// Refine grid
//---------------------------------------------------------------------
template <int dim>
void Step12<dim>::refine_grid (Vector<double> refinement_soln)
{
    // std::cout << "Refining grid ...\n";
    Vector<float> refinement_indicators (triangulation.n_active_cells());


    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    std::vector<unsigned int> dofs (dofs_per_cell);

    const QMidpoint<dim>  quadrature_formula;

    const UpdateFlags update_flags = update_gradients | update_hessians;

    FEValues<dim> fe_v (dof_handler.get_fe(), quadrature_formula, update_flags);
    std::vector<Tensor<1,dim> > dU (1);
    std::vector<Tensor<2,dim> > ddU (1);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
    {
        fe_v.reinit(cell);
        fe_v.get_function_grads(refinement_soln, dU);
        fe_v.get_function_hessians(refinement_soln, ddU);
        double h = cell->diameter ();
        double numer = 0.0;
        double denom = 0.0;
        for(int d1 = 0; d1 < dim; d1++)
        {
            for(int d2 = 0; d2 < dim; d2++)
            {
                numer += std::pow(ddU[0][d1][d2] * ddU[0][d1][d2], 2);
                denom += std::pow(dU[0] * dU[0] + epsilon * std::fabs(ddU[0][d1][d2] * ddU[0][d1][d2]), 2);
            }
        }
        if(std::fabs(numer) < 0.00001 && std::fabs(denom) < 0.00001)
        {
            refinement_indicators(cell_no) = 0.0;
        }
        else
        {
            refinement_indicators(cell_no) = std::sqrt(numer / denom);
        }
    }

    cell = dof_handler.begin_active();
    endc = dof_handler.end();
    for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
    {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();
        if ((cell->level() < max_level) &&
            (std::fabs(refinement_indicators(cell_no)) > chi_refine))
        {
            cell->set_refine_flag();
        }
        else if ((cell->level() > min_level) &&
            (std::fabs(refinement_indicators(cell_no)) < chi_coarse))
        {
            cell->set_coarsen_flag();
        }
    }

    SolutionTransfer<dim, Vector<double> > soltrans(dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement(solution);
    triangulation.execute_coarsening_and_refinement ();


    dof_handler.distribute_dofs (fe);
    solution_old.reinit(dof_handler.n_dofs());
    predictor.reinit(dof_handler.n_dofs());
    soltrans.interpolate(solution, solution_old);
    soltrans.clear();
    solution.reinit(dof_handler.n_dofs());
    solution = solution_old;
    inv_mass_matrix.reinit(dof_handler.n_dofs());
    assemble_mass_matrix();
    right_hand_side.reinit(dof_handler.n_dofs());
}

//------------------------------------------------------------------------------
// Save results to file
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::output_results (const unsigned int cycle, 
                                  const double time) const
{
    ExactSolution<dim> es(time);
    Vector<double> exact_solution;
    exact_solution.reinit(dof_handler.n_dofs());
    VectorTools::interpolate(dof_handler, es, exact_solution);

    // Output of the solution in
    std::string filename = "sol-" + Utilities::int_to_string(cycle,5) + ".vtk";
    /* std::cout << "Writing solution to <" << filename << ">" << std::endl; */
    std::ofstream outfile (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    data_out.add_data_vector(exact_solution, "exact");

    const int patch_division_ratio = 1;
    data_out.build_patches(patch_division_ratio);

    data_out.write_vtk (outfile);
}

template <int dim>
void Step12<dim>::compute_error(const double time) const
{
    ExactSolution<dim> es(time);
    Vector<double> error;
    VectorTools::integrate_difference(dof_handler, solution, es, error,
                                      QGaussLobatto<dim>(degree + 1),
                                      VectorTools::NormType::L2_norm);
    const double total_local_error = error.l2_norm();
    const double total_global_error = std::sqrt(Utilities::MPI::sum(
                  total_local_error * total_local_error, MPI_COMM_WORLD));
    std::cout << "Total L2 error: " << total_global_error << std::endl;
}

//------------------------------------------------------------------------------
// Actual computation starts from here
//------------------------------------------------------------------------------
template <int dim>
void Step12<dim>::run ()
{
   GridGenerator::hyper_ball(triangulation);
   static const HyperBallBoundary<dim> boundary;
   triangulation.set_boundary (0, boundary);
   triangulation.refine_global(min_level);
   
   std::cout << "Number of active cells:       "
             << triangulation.n_active_cells()
             << std::endl;
   
   setup_system ();
   
   std::cout << "Number of degrees of freedom: "
             << dof_handler.n_dofs()
             << std::endl;
   
   assemble_mass_matrix ();
   set_initial_condition ();
   const int initial_refinements = 2;
   for(int i = 0; i < initial_refinements; i++)
   {
       refine_grid(solution);
   }
   set_initial_condition ();
   output_results(0, 0.0);
   solve ();
}


template class Step12<2>;
